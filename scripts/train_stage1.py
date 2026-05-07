import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets import PatchDataset, build_stage1_eval_transform, build_stage1_train_transform
from src.losses import BCEDiceLoss
from src.mining import build_stage1_replay_rows, save_stage1_replay_outputs
from src.model import build_model
from src.trainer import EarlyStopper, build_optimizer, build_scheduler, save_checkpoint, train_one_epoch, validate_stage1
from src.utils import ensure_dir, load_yaml, read_csv_rows, seed_worker, set_seed, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Train stage1 patch model")
    parser.add_argument("--config", type=str, default="configs/stage1.yaml", help="Path to config file")
    parser.add_argument("--fold", type=int, default=None, help="Override fold in config")
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_device(cfg):
    device_text = str(cfg.get("device", "auto")).strip().lower()

    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_text == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    return torch.device(device_text)


def apply_fold_overrides(cfg, fold):
    cfg = dict(cfg)
    cfg["fold"] = int(fold)

    if "train_index_path_template" in cfg:
        cfg["train_index_path"] = str(cfg["train_index_path_template"]).format(fold=fold)

    if "val_index_path_template" in cfg:
        cfg["val_index_path"] = str(cfg["val_index_path_template"]).format(fold=fold)

    if "save_dir_template" in cfg:
        cfg["save_dir"] = str(cfg["save_dir_template"]).format(fold=fold)

    return cfg


def infer_patch_family(row):
    patch_family = str(row.get("patch_family", "")).strip()
    if patch_family != "":
        return patch_family

    patch_type = str(row.get("patch_type", "")).strip()

    if patch_type.startswith("positive"):
        return "positive"
    if patch_type in {"near_miss_negative", "hard_negative"}:
        return "defect_negative"
    if patch_type == "normal_negative":
        return "normal_negative"
    return "unknown"


def count_patch_families(rows):
    counter = Counter()
    for row in rows:
        counter[infer_patch_family(row)] += 1
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def build_stage1_sampler(train_rows, cfg, generator):
    sampler_mode = str(cfg.get("stage1_sampler_mode", "shuffle")).strip().lower()
    if sampler_mode != "balanced":
        return None

    desired_ratios = {
        "positive": float(cfg.get("stage1_positive_ratio", 0.50)),
        "defect_negative": float(cfg.get("stage1_defect_negative_ratio", 0.25)),
        "normal_negative": float(cfg.get("stage1_normal_ratio", 0.25)),
        "replay": float(cfg.get("stage1_replay_ratio", 0.0)),
    }

    family_counts = Counter(infer_patch_family(row) for row in train_rows)
    weights = []

    for row in train_rows:
        family = infer_patch_family(row)

        family_count = max(1, family_counts[family])
        family_ratio = desired_ratios.get(family, 0.0)

        if family_ratio <= 0.0:
            if family == "replay":
                weights.append(0.0)
                continue
            family_ratio = 1.0 / max(1, len(train_rows))

        weight = family_ratio / family_count
        weights.append(weight)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_rows),
        replacement=True,
        generator=generator,
    )


def build_stage1_loaders_from_rows(train_rows, val_rows, cfg, device):
    patch_out_size = int(cfg.get("patch_out_size", 384))
    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 8))
    seed = int(cfg.get("seed", 42))

    train_dataset = PatchDataset(
        train_rows,
        transform=build_stage1_train_transform(patch_out_size, cfg=cfg),
    )
    val_dataset = PatchDataset(
        val_rows,
        transform=build_stage1_eval_transform(patch_out_size, cfg=cfg),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = device.type == "cuda"
    worker_init_fn = seed_worker if num_workers > 0 else None
    persistent_workers = num_workers > 0
    sampler = build_stage1_sampler(train_rows, cfg, generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, len(train_rows), len(val_rows)


def build_stage1_loaders(cfg, device):
    train_rows = read_csv_rows(resolve_path(cfg["train_index_path"]))
    val_rows = read_csv_rows(resolve_path(cfg["val_index_path"]))
    return build_stage1_loaders_from_rows(train_rows, val_rows, cfg, device)


def build_scaler(cfg, device):
    amp_enabled = bool(cfg.get("amp", True))
    use_amp = amp_enabled and device.type == "cuda"

    if not use_amp:
        return None

    return torch.amp.GradScaler(device="cuda", enabled=True)


def save_history_csv(path, history_rows):
    fieldnames = [
        "epoch",
        "encoder_frozen",
        "train_loss",
        "val_loss",
        "patch_dice_all",
        "patch_dice_pos_only",
        "positive_patch_recall",
        "negative_patch_fpr",
        "stage1_score",
        "train_patch_count",
        "replay_patch_count",
        "train_family_counts_json",
        "patch_dice_by_type_json",
        "count_by_type_json",
        "lr_encoder",
        "lr_decoder",
    ]
    write_csv_rows(path, history_rows, fieldnames)


def stage1_result_is_better(current_result, best_result):
    if best_result is None:
        return True

    if float(current_result["patch_dice_pos_only"]) > float(best_result["patch_dice_pos_only"]):
        return True

    if (
        float(current_result["patch_dice_pos_only"]) == float(best_result["patch_dice_pos_only"])
        and float(current_result["negative_patch_fpr"]) < float(best_result["negative_patch_fpr"])
    ):
        return True

    if (
        float(current_result["patch_dice_pos_only"]) == float(best_result["patch_dice_pos_only"])
        and float(current_result["negative_patch_fpr"]) == float(best_result["negative_patch_fpr"])
        and float(current_result["positive_patch_recall"]) > float(best_result["positive_patch_recall"])
    ):
        return True

    return False


def should_refresh_stage1_replay(epoch, cfg):
    if not bool(cfg.get("stage1_use_replay", False)):
        return False

    if float(cfg.get("stage1_replay_ratio", 0.0)) <= 0.0:
        return False

    warmup_epochs = int(cfg.get("stage1_replay_warmup_epochs", 3))
    refresh_every = max(1, int(cfg.get("stage1_replay_refresh_every", 3)))

    if int(epoch) < warmup_epochs:
        return False

    return (int(epoch) - warmup_epochs) % refresh_every == 0


def main():
    args = parse_args()

    cfg = load_yaml(resolve_path(args.config))
    fold = int(args.fold if args.fold is not None else cfg.get("fold", 0))
    cfg = apply_fold_overrides(cfg, fold)

    set_seed(int(cfg.get("seed", 42)))

    device = build_device(cfg)
    base_train_rows = read_csv_rows(resolve_path(cfg["train_index_path"]))
    val_rows = read_csv_rows(resolve_path(cfg["val_index_path"]))
    train_loader, val_loader, train_patch_count, val_patch_count = build_stage1_loaders_from_rows(
        base_train_rows,
        val_rows,
        cfg,
        device,
    )

    model = build_model(pretrained=bool(cfg.get("pretrained", True)))
    model.to(device)

    criterion = BCEDiceLoss(
        bce_weight=float(cfg.get("bce_weight", 0.5)),
        dice_weight=float(cfg.get("dice_weight", 0.5)),
        pos_weight=float(cfg.get("pos_weight", 12.0)),
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = build_scaler(cfg, device)

    early_stopper = EarlyStopper(
        patience=int(cfg.get("early_stop_patience", 12)),
        mode="max",
        min_delta=float(cfg.get("early_stop_min_delta", 0.0)),
    )

    epochs = int(cfg.get("epochs", 30))
    freeze_encoder_epochs = int(cfg.get("freeze_encoder_epochs", 3))
    eval_threshold = float(cfg.get("stage1_eval_threshold", 0.5))

    save_dir = resolve_path(cfg["save_dir"])
    ensure_dir(save_dir)

    best_ckpt_path = save_dir / "best_stage1.pt"
    last_ckpt_path = save_dir / "last_stage1.pt"
    history_path = save_dir / "history.csv"

    history_rows = []
    best_stage1_result = None
    replay_rows = []
    replay_summary = {"total_count": 0, "count_by_type": {}, "count_by_source_family": {}}

    print(f"fold = {fold}")
    print(f"device = {device}")
    print(f"base_train_patch_count = {len(base_train_rows)}")
    print(f"val_patch_count = {val_patch_count}")
    print(f"save_dir = {save_dir}")

    for epoch_index in range(epochs):
        epoch = epoch_index + 1
        encoder_trainable = epoch_index >= freeze_encoder_epochs
        model.set_encoder_trainable(encoder_trainable)

        current_train_rows = list(base_train_rows)
        if len(replay_rows) > 0:
            current_train_rows.extend(replay_rows)

        train_loader, val_loader, train_patch_count, val_patch_count = build_stage1_loaders_from_rows(
            current_train_rows,
            val_rows,
            cfg,
            device,
        )
        train_family_counts = count_patch_families(current_train_rows)

        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            progress_desc=f"Train {epoch}/{epochs}",
        )
        val_stats = validate_stage1(
            model,
            val_loader,
            criterion,
            device,
            threshold=eval_threshold,
            progress_desc=f"Val {epoch}/{epochs}",
        )

        stage1_score = float(val_stats["patch_dice_pos_only"])
        scheduler.step(stage1_score)
        should_stop, _ = early_stopper.step(stage1_score)

        current_is_best = stage1_result_is_better(val_stats, best_stage1_result)
        if current_is_best:
            best_stage1_result = {
                "patch_dice_all": float(val_stats["patch_dice_all"]),
                "patch_dice_pos_only": float(val_stats["patch_dice_pos_only"]),
                "positive_patch_recall": float(val_stats["positive_patch_recall"]),
                "negative_patch_fpr": float(val_stats["negative_patch_fpr"]),
                "patch_dice_by_type": dict(val_stats["patch_dice_by_type"]),
                "count_by_type": dict(val_stats["count_by_type"]),
                "replay_summary": replay_summary,
            }

        history_rows.append(
            {
                "epoch": epoch,
                "encoder_frozen": not encoder_trainable,
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["val_loss"],
                "patch_dice_all": val_stats["patch_dice_all"],
                "patch_dice_pos_only": val_stats["patch_dice_pos_only"],
                "positive_patch_recall": val_stats["positive_patch_recall"],
                "negative_patch_fpr": val_stats["negative_patch_fpr"],
                "stage1_score": stage1_score,
                "train_patch_count": train_patch_count,
                "replay_patch_count": len(replay_rows),
                "train_family_counts_json": json.dumps(train_family_counts, ensure_ascii=False, sort_keys=True),
                "patch_dice_by_type_json": json.dumps(val_stats["patch_dice_by_type"], ensure_ascii=False, sort_keys=True),
                "count_by_type_json": json.dumps(val_stats["count_by_type"], ensure_ascii=False, sort_keys=True),
                "lr_encoder": train_stats["lr_encoder"],
                "lr_decoder": train_stats["lr_decoder"],
            }
        )
        save_history_csv(history_path, history_rows)

        checkpoint = {
            "epoch": epoch,
            "fold": fold,
            "config": cfg,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "early_stopper_state_dict": early_stopper.state_dict(),
            "best_stage1_result": best_stage1_result,
            "replay_summary": replay_summary,
            "current_val_stats": {
                "patch_dice_all": float(val_stats["patch_dice_all"]),
                "patch_dice_pos_only": float(val_stats["patch_dice_pos_only"]),
                "positive_patch_recall": float(val_stats["positive_patch_recall"]),
                "negative_patch_fpr": float(val_stats["negative_patch_fpr"]),
                "patch_dice_by_type": dict(val_stats["patch_dice_by_type"]),
                "count_by_type": dict(val_stats["count_by_type"]),
            },
        }

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        save_checkpoint(last_ckpt_path, checkpoint)

        if current_is_best:
            save_checkpoint(best_ckpt_path, checkpoint)

        if should_refresh_stage1_replay(epoch, cfg):
            replay_rows, replay_summary = build_stage1_replay_rows(
                model=model,
                base_patch_rows=base_train_rows,
                cfg=cfg,
                device=device,
                source_epoch=epoch,
            )
            save_stage1_replay_outputs(save_dir, epoch, replay_rows, replay_summary)
        else:
            replay_summary = {
                **replay_summary,
                "source_epoch": replay_summary.get("source_epoch", ""),
                "total_count": len(replay_rows),
            }

        print(
            f"epoch {epoch}/{epochs} | "
            f"encoder_frozen={not encoder_trainable} | "
            f"train_patches={train_patch_count} | "
            f"replay_patches={len(replay_rows)} | "
            f"train_loss={train_stats['loss']:.6f} | "
            f"val_loss={val_stats['val_loss']:.6f} | "
            f"patch_dice_all={val_stats['patch_dice_all']:.6f} | "
            f"patch_dice_pos_only={val_stats['patch_dice_pos_only']:.6f} | "
            f"positive_recall={val_stats['positive_patch_recall']:.6f} | "
            f"negative_fpr={val_stats['negative_patch_fpr']:.6f} | "
            f"lr_encoder={train_stats['lr_encoder']:.6e} | "
            f"lr_decoder={train_stats['lr_decoder']:.6e}"
        )

        if should_stop:
            print("early stopping triggered, stage1 stopped early.")
            break

    print("stage1 training finished.")


if __name__ == "__main__":
    main()
