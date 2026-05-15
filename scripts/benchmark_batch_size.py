import argparse
import copy
import gc
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets import (
    PatchDataset,
    ROIDataset,
    build_same_size_batch_sampler,
    build_stage1_train_transform,
    build_stage2_train_transform,
    normalize_size,
    resolve_stage2_image_size,
    stage2_uses_native_size,
)
from src.losses import BCEDiceLoss
from src.model import build_model, build_model_from_config
from src.parallel import maybe_wrap_data_parallel
from src.samples import load_samples, split_samples
from src.trainer import build_amp_grad_scaler, build_optimizer, train_one_epoch
from src.utils import load_stage_config, read_csv_rows, seed_worker, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark max training batch size with real data")
    parser.add_argument("--config", type=str, default="configs/canonical_baseline.yaml")
    parser.add_argument("--stage", choices=["stage1", "stage2"], default="stage2")
    parser.add_argument("--batches", type=int, default=2, help="Number of training batches to run per candidate")
    parser.add_argument("--candidates", type=str, default="", help="Comma-separated global batch sizes")
    parser.add_argument("--model-variant", type=str, default="", help="Override stage2 model_variant")
    parser.add_argument("--skip-attention-levels", type=str, default="", help="Comma-separated skip levels for skipgate")
    parser.add_argument("--synthetic", action="store_true", help="Use random tensors instead of reading dataset images")
    parser.add_argument("--synthetic-size", type=str, default="", help="Synthetic HxW size for native Stage2 benchmarks")
    parser.add_argument("--synthetic-mask-mode", choices=["sparse", "empty"], default="sparse")
    parser.add_argument("--normal-fp-loss-weight", type=float, default=None)
    parser.add_argument("--normal-fp-topk-ratio", type=float, default=None)
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_candidates(text, defaults):
    text = str(text or "").strip()
    if text == "":
        return list(defaults)
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def limit_loader(loader, max_batches):
    for index, batch in enumerate(loader):
        if index >= int(max_batches):
            break
        yield batch


def build_stage2_rows(cfg):
    sample_rows = load_samples(cfg["samples_path"], PROJECT_ROOT)
    defect_train_rows, _, normal_train_rows, _ = split_samples(sample_rows)
    rows = list(defect_train_rows) + list(normal_train_rows)
    return rows


def build_stage1_rows(cfg):
    return read_csv_rows(resolve_path(cfg["train_index_path"]))


def make_loader(rows, cfg, stage, device, batch_size):
    num_workers = int(cfg.get("num_workers", 8))
    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("seed", 42)))

    if stage == "stage1":
        out_size = int(cfg.get("patch_out_size", 384))
        dataset = PatchDataset(
            rows,
            transform=build_stage1_train_transform(out_size, cfg=cfg),
        )
    else:
        image_size = resolve_stage2_image_size(cfg)
        dataset = ROIDataset(
            rows,
            image_size=image_size,
            transform=build_stage2_train_transform(image_size, cfg=cfg),
        )

        if stage2_uses_native_size(cfg):
            batch_sampler = build_same_size_batch_sampler(
                rows=rows,
                cfg=cfg,
                default_batch_size=batch_size,
                shuffle=True,
                seed=int(cfg.get("seed", 42)),
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
                worker_init_fn=seed_worker if num_workers > 0 else None,
                generator=generator,
                persistent_workers=num_workers > 0,
            )

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def make_model(cfg, stage, device):
    if stage == "stage1":
        model = build_model(
            pretrained=bool(cfg.get("pretrained", True)),
            allow_pretrained_fallback=bool(cfg.get("allow_pretrained_fallback", True)),
        )
    else:
        model = build_model_from_config(cfg)
    model.to(device)
    return model


def make_criterion(cfg, stage, device):
    del device
    if stage == "stage1":
        return BCEDiceLoss(
            bce_weight=float(cfg.get("bce_weight", 0.5)),
            dice_weight=float(cfg.get("dice_weight", 0.5)),
            pos_weight=float(cfg.get("pos_weight", 12.0)),
        )
    return BCEDiceLoss(
        bce_weight=float(cfg.get("bce_weight", 0.5)),
        dice_weight=float(cfg.get("dice_weight", 0.5)),
        pos_weight=float(cfg.get("pos_weight", 12.0)),
        normal_fp_loss_weight=float(cfg.get("normal_fp_loss_weight", 0.0)),
        normal_fp_topk_ratio=float(cfg.get("normal_fp_topk_ratio", 1.0)),
        deep_supervision_weight=float(cfg.get("deep_supervision_weight", 0.0)),
        deep_supervision_decay=float(cfg.get("deep_supervision_decay", 0.5)),
        boundary_aux_weight=float(cfg.get("boundary_aux_weight", 0.0)),
        boundary_width=int(cfg.get("boundary_width", 3)),
    )


def cuda_memory_summary():
    if not torch.cuda.is_available():
        return {}
    return {
        f"gpu{index}_max_allocated_gb": round(torch.cuda.max_memory_allocated(index) / (1024 ** 3), 3)
        for index in range(torch.cuda.device_count())
    }


def synthetic_hw(cfg, stage):
    if stage == "stage1":
        size = int(cfg.get("patch_out_size", 384))
        return size, size

    synthetic_size = str(cfg.get("_synthetic_size", "")).strip()
    if synthetic_size != "":
        return normalize_size(synthetic_size)

    image_size = resolve_stage2_image_size(cfg)
    if image_size is None:
        raise ValueError("native Stage2 synthetic benchmark needs --synthetic-size HxW, for example 2160x3840")

    return normalize_size(image_size)


def benchmark_candidate(base_cfg, rows, stage, batch_size, max_batches):
    cfg = copy.deepcopy(base_cfg)
    cfg["batch_size"] = int(batch_size)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "auto")).lower() != "cpu" else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = None
    loader = None
    optimizer = None
    scaler = None

    try:
        model = make_model(cfg, stage, device)
        criterion = make_criterion(cfg, stage, device)
        optimizer = build_optimizer(model, cfg)
        scaler = build_amp_grad_scaler(device, enabled=bool(cfg.get("amp", True)))
        model, parallel_summary = maybe_wrap_data_parallel(model, cfg, device)
        if cfg.get("_synthetic_benchmark", False):
            height, width = synthetic_hw(cfg, stage)
            if cfg.get("_synthetic_mask_mode", "sparse") == "empty":
                mask = torch.zeros(int(batch_size), 1, height, width)
            else:
                mask = (torch.rand(int(batch_size), 1, height, width) > 0.98).float()
            loader = [
                {
                    "image": torch.rand(int(batch_size), 3, height, width),
                    "mask": mask.clone(),
                }
                for _ in range(int(max_batches))
            ]
            stats = train_one_epoch(
                model=model,
                loader=loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                progress_desc=None,
            )
        else:
            loader = make_loader(rows, cfg, stage, device, batch_size=batch_size)
            stats = train_one_epoch(
                model=model,
                loader=limit_loader(loader, max_batches),
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                progress_desc=None,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "ok": True,
            "batch_size": int(batch_size),
            "loss": float(stats["loss"]),
            "parallel": parallel_summary,
            **cuda_memory_summary(),
        }
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" not in message.lower():
            raise
        return {
            "ok": False,
            "batch_size": int(batch_size),
            "error": "CUDA out of memory",
            **cuda_memory_summary(),
        }
    finally:
        del loader
        del optimizer
        del scaler
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    cfg = load_stage_config(resolve_path(args.config), args.stage)
    if args.model_variant:
        cfg["model_variant"] = args.model_variant
        cfg["model_name"] = args.model_variant
        if args.model_variant != "resnet34_unet_baseline":
            cfg["stage1_load_strict"] = False
    if args.skip_attention_levels:
        cfg["skip_attention_levels"] = [item.strip() for item in args.skip_attention_levels.split(",") if item.strip()]
    if args.synthetic:
        cfg["_synthetic_benchmark"] = True
        cfg["_synthetic_size"] = args.synthetic_size
        cfg["_synthetic_mask_mode"] = args.synthetic_mask_mode
    if args.normal_fp_loss_weight is not None:
        cfg["normal_fp_loss_weight"] = float(args.normal_fp_loss_weight)
    if args.normal_fp_topk_ratio is not None:
        cfg["normal_fp_topk_ratio"] = float(args.normal_fp_topk_ratio)

    if args.stage == "stage1":
        rows = build_stage1_rows(cfg)
        defaults = [128, 192, 256, 320, 384, 448, 512]
    else:
        rows = build_stage2_rows(cfg)
        defaults = [48, 64, 96, 128, 160, 192, 224, 256]

    candidates = parse_candidates(args.candidates, defaults)
    print(
        {
            "stage": args.stage,
            "model_variant": cfg.get("model_variant", "stage1_baseline"),
            "rows": len(rows),
            "candidates": candidates,
            "batches_per_candidate": int(args.batches),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "synthetic": bool(args.synthetic),
        }
    )

    for batch_size in candidates:
        result = benchmark_candidate(cfg, rows, args.stage, batch_size, args.batches)
        print(result, flush=True)
        if not result["ok"]:
            break


if __name__ == "__main__":
    main()
