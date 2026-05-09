import argparse
import json
import sys
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets import ROIDataset, build_stage2_eval_transform, build_stage2_train_transform
from src.losses import BCEDiceLoss
from src.metrics import compare_stage2_results
from src.mining import (
    build_hard_normal_pool,
    resolve_hard_normal_count,
    sample_rows_with_frame_gap,
    save_stage2_hard_normal_outputs,
)
from src.model import build_model
from src.samples import load_samples, split_samples_for_fold
from src.trainer import EarlyStopper, build_amp_grad_scaler, build_optimizer, build_scheduler, load_checkpoint, save_checkpoint, train_one_epoch, validate_stage2
from src.utils import ensure_dir, load_yaml, read_csv_rows, seed_worker, set_seed, write_csv_rows
from scripts.evaluate_val import evaluate_and_save_stage2


def parse_args():
    parser = argparse.ArgumentParser(description="Train stage2 full-image segmentation model")
    parser.add_argument("--config", type=str, default="configs/stage2.yaml", help="Path to config file")
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

    template_keys = [
        ("defect_train_manifest_template", "defect_train_manifest"),
        ("defect_val_manifest_template", "defect_val_manifest"),
        ("normal_train_manifest_template", "normal_train_manifest"),
        ("normal_val_manifest_template", "normal_val_manifest"),
        ("stage1_checkpoint_template", "stage1_checkpoint"),
        ("save_dir_template", "save_dir"),
    ]

    for template_key, target_key in template_keys:
        if template_key in cfg:
            cfg[target_key] = str(cfg[template_key]).format(fold=fold)

    return cfg


def build_scaler(cfg, device):
    amp_enabled = bool(cfg.get("amp", True))
    use_amp = amp_enabled and device.type == "cuda"

    if not use_amp:
        return None

    return build_amp_grad_scaler(device, enabled=True)


def sample_normal_rows(normal_rows, k, seed):
    return sample_rows_with_frame_gap(
        normal_rows,
        k=k,
        seed=seed,
        min_frame_gap=0,
    )


def count_rows_by_device(rows):
    counter = {}
    for row in rows:
        device = str(row.get("device", "")).strip() or "unknown"
        counter[device] = counter.get(device, 0) + 1
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def stage2_row_key(row):
    return (
        str(row.get("sample_id", "")).strip(),
        str(row.get("image_name", "")).strip(),
        str(row.get("image_path", "")).strip(),
    )


def sample_hard_normal_rows(hard_normal_rows, k, seed, max_repeats_per_row=2):
    if int(k) <= 0 or len(hard_normal_rows) == 0:
        return []

    max_repeats_per_row = max(1, int(max_repeats_per_row))
    max_sample_count = len(hard_normal_rows) * max_repeats_per_row
    k = min(int(k), max_sample_count)

    selected_rows = sample_rows_with_frame_gap(
        hard_normal_rows,
        k=k,
        seed=seed,
        min_frame_gap=0,
        score_key="hard_score",
        descending=True,
    )

    if len(selected_rows) >= k:
        return selected_rows

    refill_pool = selected_rows or sorted(
        list(hard_normal_rows),
        key=lambda row: (float(row.get("hard_score", 0.0)), stage2_row_key(row)),
        reverse=True,
    )
    rng = random.Random(seed + 1009)

    while len(selected_rows) < int(k) and len(refill_pool) > 0:
        shuffled_pool = list(refill_pool)
        rng.shuffle(shuffled_pool)
        for row in shuffled_pool:
            row_count = sum(1 for selected_row in selected_rows if stage2_row_key(selected_row) == stage2_row_key(row))
            if row_count >= max_repeats_per_row:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= int(k):
                break

    return selected_rows


def should_refresh_hard_normal(epoch, cfg):
    if not bool(cfg.get("use_hard_normal_replay", False)):
        return False

    if float(cfg.get("stage2_hard_normal_ratio", 0.0)) <= 0.0 and float(cfg.get("hard_normal_k_factor", 0.0)) <= 0.0:
        return False

    warmup_epochs = int(cfg.get("hard_normal_warmup_epochs", 2))
    refresh_every = max(1, int(cfg.get("hard_normal_refresh_every", 2)))

    if int(epoch) < warmup_epochs:
        return False

    return (int(epoch) - warmup_epochs) % refresh_every == 0


def build_epoch_train_rows(defect_train_rows, normal_train_rows, hard_normal_rows, epoch_seed, cfg):
    defect_train_rows = list(defect_train_rows)
    normal_train_rows = list(normal_train_rows)
    hard_normal_rows = list(hard_normal_rows)

    random_normal_k_factor = float(cfg.get("random_normal_k_factor", 1.0))
    target_normal_budget_count = int(round(len(defect_train_rows) * random_normal_k_factor))
    hard_normal_max_repeats = max(1, int(cfg.get("hard_normal_max_repeats_per_epoch", 2)))

    if bool(cfg.get("use_hard_normal_replay", False)) and len(hard_normal_rows) > 0:
        requested_hard_normal_count = min(
            target_normal_budget_count,
            resolve_hard_normal_count(
                defect_count=len(defect_train_rows),
                random_normal_count=target_normal_budget_count,
                cfg=cfg,
            ),
        )
        target_hard_normal_count = min(
            requested_hard_normal_count,
            len(hard_normal_rows) * hard_normal_max_repeats,
        )
    else:
        requested_hard_normal_count = 0
        target_hard_normal_count = 0

    target_random_normal_count = max(0, target_normal_budget_count - target_hard_normal_count)
    sampled_hard_normal_rows = sample_hard_normal_rows(
        hard_normal_rows=hard_normal_rows,
        k=target_hard_normal_count,
        seed=epoch_seed + 97,
        max_repeats_per_row=hard_normal_max_repeats,
    )
    hard_normal_keys = {stage2_row_key(row) for row in sampled_hard_normal_rows}
    random_candidate_rows = [
        row
        for row in normal_train_rows
        if stage2_row_key(row) not in hard_normal_keys
    ]
    if len(random_candidate_rows) < target_random_normal_count:
        random_candidate_rows = normal_train_rows
    sampled_normal_rows = sample_normal_rows(
        normal_rows=random_candidate_rows,
        k=target_random_normal_count,
        seed=epoch_seed,
    )

    epoch_rows = defect_train_rows + sampled_normal_rows + sampled_hard_normal_rows

    rng = random.Random(epoch_seed)
    rng.shuffle(epoch_rows)
    return epoch_rows, {
        "normal_budget_count": target_normal_budget_count,
        "requested_hard_normal_count": requested_hard_normal_count,
        "target_random_normal_count": target_random_normal_count,
        "target_hard_normal_count": target_hard_normal_count,
        "random_normal_count": len(sampled_normal_rows),
        "hard_normal_count": len(sampled_hard_normal_rows),
        "hard_pool_size": len(hard_normal_rows),
        "hard_normal_max_repeats_per_epoch": hard_normal_max_repeats,
        "train_device_counts": count_rows_by_device(epoch_rows),
    }


def build_stage2_train_loader(rows, cfg, device):
    image_size = int(cfg.get("image_size", 640))
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 8))
    seed = int(cfg.get("seed", 42))

    dataset = ROIDataset(
        rows,
        image_size=image_size,
        transform=build_stage2_train_transform(image_size, cfg=cfg),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = device.type == "cuda"
    worker_init_fn = seed_worker if num_workers > 0 else None
    persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=persistent_workers,
    )


def build_stage2_val_loader(defect_val_rows, normal_val_rows, cfg, device):
    image_size = int(cfg.get("image_size", 640))
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 8))
    seed = int(cfg.get("seed", 42))

    rows = list(defect_val_rows) + list(normal_val_rows)

    dataset = ROIDataset(
        rows,
        image_size=image_size,
        transform=build_stage2_eval_transform(image_size, cfg=cfg),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = device.type == "cuda"
    worker_init_fn = seed_worker if num_workers > 0 else None
    persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=persistent_workers,
    )


def build_threshold_grid(cfg):
    if "threshold_grid" in cfg:
        return [float(item) for item in cfg["threshold_grid"]]

    start = float(cfg.get("threshold_grid_start", 0.10))
    end = float(cfg.get("threshold_grid_end", 0.90))
    step = float(cfg.get("threshold_grid_step", 0.02))
    if step <= 0:
        raise ValueError("threshold_grid_step must be positive")

    values = []
    current = start

    while current <= end + 1e-8:
        values.append(round(current, 6))
        current += step

    if not values or abs(values[-1] - end) > 1e-8:
        values.append(round(end, 6))

    return values


def build_min_area_grid(cfg):
    values = cfg.get("min_area_grid", [0, 8, 16, 24, 32, 48])
    return [int(item) for item in values]


def load_stage2_rows(cfg, fold):
    samples_path = str(cfg.get("samples_path", "")).strip()
    if samples_path != "":
        sample_rows = load_samples(samples_path, PROJECT_ROOT)
        return split_samples_for_fold(sample_rows, fold=fold)

    defect_train_rows = read_csv_rows(resolve_path(cfg["defect_train_manifest"]))
    defect_val_rows = read_csv_rows(resolve_path(cfg["defect_val_manifest"]))
    normal_train_rows = read_csv_rows(resolve_path(cfg["normal_train_manifest"]))
    normal_val_rows = read_csv_rows(resolve_path(cfg["normal_val_manifest"]))
    return defect_train_rows, defect_val_rows, normal_train_rows, normal_val_rows


def save_history_csv(path, history_rows):
    fieldnames = [
        "epoch",
        "train_loss",
        "defect_dice",
        "defect_iou",
        "defect_image_recall",
        "normal_fpr",
        "normal_count",
        "normal_fp_count",
        "normal_fp_pixel_sum",
        "normal_fp_pixel_mean",
        "normal_fp_pixel_median",
        "normal_fp_pixel_p95",
        "normal_largest_fp_area_mean",
        "normal_largest_fp_area_median",
        "normal_largest_fp_area_p95",
        "normal_largest_fp_area_max",
        "threshold",
        "min_area",
        "stage2_score",
        "normal_budget_count",
        "requested_hard_normal_count",
        "target_random_normal_count",
        "target_hard_normal_count",
        "random_normal_count",
        "hard_normal_count",
        "hard_pool_size",
        "hard_normal_max_repeats_per_epoch",
        "train_device_counts_json",
        "lr_encoder",
        "lr_decoder",
    ]
    write_csv_rows(path, history_rows, fieldnames)


STAGE2_SUMMARY_KEYS = [
    "defect_dice",
    "defect_iou",
    "defect_image_recall",
    "normal_fpr",
    "normal_count",
    "normal_fp_count",
    "normal_fp_pixel_sum",
    "normal_fp_pixel_mean",
    "normal_fp_pixel_median",
    "normal_fp_pixel_p95",
    "normal_largest_fp_area_mean",
    "normal_largest_fp_area_median",
    "normal_largest_fp_area_p95",
    "normal_largest_fp_area_max",
    "threshold",
    "min_area",
    "stage2_score",
]


def summarize_stage2_result(val_stats):
    summary = {}
    for key in STAGE2_SUMMARY_KEYS:
        if key in {"normal_count", "normal_fp_count", "min_area"}:
            summary[key] = int(val_stats[key])
        else:
            summary[key] = float(val_stats[key])
    return summary


def main():
    args = parse_args()

    cfg = load_yaml(resolve_path(args.config))
    fold = int(args.fold if args.fold is not None else cfg.get("fold", 0))
    cfg = apply_fold_overrides(cfg, fold)

    set_seed(int(cfg.get("seed", 42)))

    device = build_device(cfg)

    defect_train_rows, defect_val_rows, normal_train_rows, normal_val_rows = load_stage2_rows(cfg, fold)

    val_loader = build_stage2_val_loader(defect_val_rows, normal_val_rows, cfg, device)

    deep_supervision_enabled = bool(cfg.get("deep_supervision_enable", False))
    boundary_aux_enabled = bool(cfg.get("boundary_aux_enable", False))
    model = build_model(
        pretrained=bool(cfg.get("pretrained", False)),
        deep_supervision=deep_supervision_enabled,
        boundary_aux=boundary_aux_enabled,
    )
    model.to(device)

    stage1_checkpoint_path = resolve_path(cfg["stage1_checkpoint"])
    load_checkpoint(
        stage1_checkpoint_path,
        model,
        map_location=device,
        strict=not (deep_supervision_enabled or boundary_aux_enabled),
    )

    criterion = BCEDiceLoss(
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

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = build_scaler(cfg, device)

    early_stopper = EarlyStopper(
        patience=int(cfg.get("early_stop_patience", 12)),
        mode="max",
        min_delta=float(cfg.get("early_stop_min_delta", 0.0)),
    )

    epochs = int(cfg.get("epochs", 25))
    target_normal_fpr = float(cfg.get("target_normal_fpr", 0.10))
    lambda_fpr_penalty = float(cfg.get("lambda_fpr_penalty", 2.0))
    train_eval_threshold = float(cfg.get("train_eval_threshold", cfg.get("threshold", 0.50)))
    train_eval_min_area = int(cfg.get("train_eval_min_area", 0))

    save_dir = resolve_path(cfg["save_dir"])
    ensure_dir(save_dir)

    best_ckpt_path = save_dir / "best_stage2.pt"
    last_ckpt_path = save_dir / "last_stage2.pt"
    history_path = save_dir / "history.csv"

    history_rows = []
    best_stage2_result = None
    hard_normal_rows = []
    hard_normal_summary = {"pool_size": 0, "count_by_device": {}, "top_score": 0.0}

    print(f"fold = {fold}")
    print(f"device = {device}")
    print(f"stage1_checkpoint = {stage1_checkpoint_path}")
    print(f"defect_train_count = {len(defect_train_rows)}")
    print(f"defect_val_count = {len(defect_val_rows)}")
    print(f"normal_train_count = {len(normal_train_rows)}")
    print(f"normal_val_count = {len(normal_val_rows)}")
    print(f"train_eval_threshold = {train_eval_threshold}")
    print(f"train_eval_min_area = {train_eval_min_area}")
    print(f"save_dir = {save_dir}")

    try:
        for epoch_index in range(epochs):
            epoch = epoch_index + 1
            epoch_seed = int(cfg.get("seed", 42)) + epoch

            epoch_train_rows, epoch_sampling_summary = build_epoch_train_rows(
                defect_train_rows,
                normal_train_rows,
                hard_normal_rows,
                epoch_seed,
                cfg,
            )
            train_loader = build_stage2_train_loader(epoch_train_rows, cfg, device)

            train_stats = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                progress_desc=f"Train {epoch}/{epochs}",
            )

            val_stats = validate_stage2(
                model=model,
                loader=val_loader,
                device=device,
                threshold=train_eval_threshold,
                min_area=train_eval_min_area,
                target_normal_fpr=target_normal_fpr,
                lambda_fpr_penalty=lambda_fpr_penalty,
                progress_desc=f"Val {epoch}/{epochs}",
                amp_enabled=bool(cfg.get("amp", True)),
            )

            stage2_score = float(val_stats["stage2_score"])
            scheduler.step(stage2_score)
            should_stop, _ = early_stopper.step(stage2_score)

            current_is_best = compare_stage2_results(
                val_stats,
                best_stage2_result,
                target_normal_fpr=target_normal_fpr,
            )

            if current_is_best:
                best_stage2_result = {
                    **summarize_stage2_result(val_stats),
                    "hard_normal_summary": hard_normal_summary,
                }

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_stats["loss"],
                    "defect_dice": val_stats["defect_dice"],
                    "defect_iou": val_stats["defect_iou"],
                    "defect_image_recall": val_stats["defect_image_recall"],
                    "normal_fpr": val_stats["normal_fpr"],
                    "normal_count": val_stats["normal_count"],
                    "normal_fp_count": val_stats["normal_fp_count"],
                    "normal_fp_pixel_sum": val_stats["normal_fp_pixel_sum"],
                    "normal_fp_pixel_mean": val_stats["normal_fp_pixel_mean"],
                    "normal_fp_pixel_median": val_stats["normal_fp_pixel_median"],
                    "normal_fp_pixel_p95": val_stats["normal_fp_pixel_p95"],
                    "normal_largest_fp_area_mean": val_stats["normal_largest_fp_area_mean"],
                    "normal_largest_fp_area_median": val_stats["normal_largest_fp_area_median"],
                    "normal_largest_fp_area_p95": val_stats["normal_largest_fp_area_p95"],
                    "normal_largest_fp_area_max": val_stats["normal_largest_fp_area_max"],
                    "threshold": val_stats["threshold"],
                    "min_area": val_stats["min_area"],
                    "stage2_score": val_stats["stage2_score"],
                    "normal_budget_count": epoch_sampling_summary["normal_budget_count"],
                    "requested_hard_normal_count": epoch_sampling_summary["requested_hard_normal_count"],
                    "target_random_normal_count": epoch_sampling_summary["target_random_normal_count"],
                    "target_hard_normal_count": epoch_sampling_summary["target_hard_normal_count"],
                    "random_normal_count": epoch_sampling_summary["random_normal_count"],
                    "hard_normal_count": epoch_sampling_summary["hard_normal_count"],
                    "hard_pool_size": epoch_sampling_summary["hard_pool_size"],
                    "hard_normal_max_repeats_per_epoch": epoch_sampling_summary["hard_normal_max_repeats_per_epoch"],
                    "train_device_counts_json": json.dumps(epoch_sampling_summary["train_device_counts"], ensure_ascii=False, sort_keys=True),
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
                "best_stage2_result": best_stage2_result,
                "hard_normal_summary": hard_normal_summary,
                "current_val_stats": summarize_stage2_result(val_stats),
            }

            if scaler is not None:
                checkpoint["scaler_state_dict"] = scaler.state_dict()

            save_checkpoint(last_ckpt_path, checkpoint)

            if current_is_best:
                save_checkpoint(best_ckpt_path, checkpoint)

            if should_refresh_hard_normal(epoch, cfg):
                hard_normal_rows, hard_normal_summary = build_hard_normal_pool(
                    model=model,
                    normal_rows=normal_train_rows,
                    cfg=cfg,
                    device=device,
                    source_epoch=epoch,
                    threshold=float(val_stats["threshold"]),
                    min_area=int(val_stats["min_area"]),
                    defect_count=len(defect_train_rows),
                    random_normal_count=epoch_sampling_summary["normal_budget_count"],
                )
                save_stage2_hard_normal_outputs(save_dir, epoch, hard_normal_rows, hard_normal_summary)
            else:
                hard_normal_summary = {
                    **hard_normal_summary,
                    "pool_size": len(hard_normal_rows),
                }

            if (
                int(epoch_sampling_summary["hard_pool_size"]) > 0
                and int(epoch_sampling_summary["target_hard_normal_count"]) > 0
                and int(epoch_sampling_summary["hard_normal_count"]) == 0
            ):
                print(
                    "Warning: hard normal pool is non-empty but no hard normal rows were sampled for this epoch."
                )
            if int(epoch_sampling_summary["target_hard_normal_count"]) < int(epoch_sampling_summary["requested_hard_normal_count"]):
                print(
                    "Warning: hard normal target was capped by hard_normal_max_repeats_per_epoch "
                    f"({epoch_sampling_summary['target_hard_normal_count']}/"
                    f"{epoch_sampling_summary['requested_hard_normal_count']})."
                )

            print(
                f"epoch {epoch}/{epochs} | "
                f"random_normal={epoch_sampling_summary['random_normal_count']} | "
                f"hard_normal={epoch_sampling_summary['hard_normal_count']}/{epoch_sampling_summary['requested_hard_normal_count']} | "
                f"hard_pool={epoch_sampling_summary['hard_pool_size']} | "
                f"train_loss={train_stats['loss']:.6f} | "
                f"defect_dice={val_stats['defect_dice']:.6f} | "
                f"defect_iou={val_stats['defect_iou']:.6f} | "
                f"defect_recall={val_stats['defect_image_recall']:.6f} | "
                f"normal_fpr={val_stats['normal_fpr']:.6f} | "
                f"normal_fp={int(val_stats['normal_fp_count'])}/{int(val_stats['normal_count'])} | "
                f"thr={val_stats['threshold']:.2f} | "
                f"min_area={val_stats['min_area']} | "
                f"stage2_score={val_stats['stage2_score']:.6f} | "
                f"lr_encoder={train_stats['lr_encoder']:.6e} | "
                f"lr_decoder={train_stats['lr_decoder']:.6e}"
            )

            if should_stop:
                print("early stopping triggered, stage2 stopped early.")
                break
    finally:
        if bool(cfg.get("auto_evaluate_after_train", True)) and best_ckpt_path.exists():
            try:
                evaluate_and_save_stage2(
                    cfg=cfg,
                    fold=fold,
                    checkpoint_path=best_ckpt_path,
                    progress_desc="Auto eval",
                )
                print("automatic validation export finished.")
            except Exception as exc:
                print(f"Warning: automatic validation export failed. Detail: {exc}")

    print("stage2 training finished.")


if __name__ == "__main__":
    main()
