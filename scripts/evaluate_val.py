import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets import (
    ROIDataset,
    build_same_size_batch_sampler,
    build_stage2_eval_transform,
    resolve_stage2_image_size,
    stage2_batch_size_by_image_size,
    stage2_uses_native_size,
)
from src.model import build_model_from_config
from src.samples import load_samples, split_samples
from src.trainer import load_checkpoint, validate_stage2
from src.utils import load_stage_config, read_csv_rows, save_json, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate best stage2 model on validation set")
    parser.add_argument("--config", type=str, default="configs/canonical_baseline.yaml", help="Path to config file")
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_threshold_grid(cfg):
    if "eval_threshold_grid" in cfg:
        return [float(item) for item in cfg["eval_threshold_grid"]]

    if "threshold_grid" in cfg:
        return [float(item) for item in cfg["threshold_grid"]]

    start = float(cfg.get("eval_threshold_grid_start", cfg.get("threshold_grid_start", 0.10)))
    end = float(cfg.get("eval_threshold_grid_end", cfg.get("threshold_grid_end", 0.90)))
    step = float(cfg.get("eval_threshold_grid_step", cfg.get("threshold_grid_step", 0.02)))
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
    values = cfg.get("eval_min_area_grid", cfg.get("min_area_grid", [0, 8, 16, 24, 32, 48]))
    return [int(item) for item in values]


def load_val_rows(cfg):
    samples_path = str(cfg.get("samples_path", "")).strip()
    if samples_path != "":
        sample_rows = load_samples(samples_path, PROJECT_ROOT)
        _, defect_val_rows, _, normal_val_rows = split_samples(sample_rows)
        return defect_val_rows + normal_val_rows

    defect_val_rows = read_csv_rows(resolve_path(cfg["defect_val_manifest"]))
    normal_val_rows = read_csv_rows(resolve_path(cfg["normal_val_manifest"]))
    return defect_val_rows + normal_val_rows


def eval_batch_size_by_image_size(cfg):
    mapping = cfg.get("eval_batch_size_by_image_size", cfg.get("stage2_eval_batch_size_by_image_size", None))
    if mapping is None:
        return stage2_batch_size_by_image_size(cfg)

    return {str(key).strip(): int(value) for key, value in dict(mapping).items()}


def evaluate_and_save_stage2(cfg, checkpoint_path=None, progress_desc=None):
    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "auto")).lower() != "cpu" else "cpu")

    rows = load_val_rows(cfg)

    image_size = resolve_stage2_image_size(cfg)
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 0))

    dataset = ROIDataset(
        rows,
        image_size=image_size,
        transform=build_stage2_eval_transform(image_size, cfg=cfg),
    )

    if stage2_uses_native_size(cfg):
        batch_sampler = build_same_size_batch_sampler(
            rows=rows,
            cfg=cfg,
            default_batch_size=batch_size,
            batch_size_by_image_size=eval_batch_size_by_image_size(cfg),
            shuffle=False,
            seed=int(cfg.get("seed", 42)),
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

    model = build_model_from_config(cfg)
    model.to(device)

    save_dir = resolve_path(cfg["save_dir"])
    if checkpoint_path is None:
        checkpoint_path = save_dir / "best_stage2.pt"
    else:
        checkpoint_path = resolve_path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)

    target_normal_fpr = float(cfg.get("target_normal_fpr", 0.10))
    lambda_fpr_penalty = float(cfg.get("lambda_fpr_penalty", 2.0))
    raw_threshold = float(cfg.get("threshold", 0.5))
    include_auprc = bool(cfg.get("eval_include_auprc", False))

    raw_result = validate_stage2(
        model=model,
        loader=loader,
        device=device,
        threshold=raw_threshold,
        min_area=0,
        target_normal_fpr=target_normal_fpr,
        lambda_fpr_penalty=lambda_fpr_penalty,
        progress_desc=None if progress_desc is None else f"{progress_desc} raw",
        include_auprc=include_auprc,
    )

    post_result = validate_stage2(
        model=model,
        loader=loader,
        device=device,
        threshold_values=build_threshold_grid(cfg),
        min_area_values=build_min_area_grid(cfg),
        target_normal_fpr=target_normal_fpr,
        lambda_fpr_penalty=lambda_fpr_penalty,
        progress_desc=None if progress_desc is None else f"{progress_desc} post",
        include_auprc=include_auprc,
    )

    checkpoint_best = checkpoint.get("best_stage2_result", {})

    metrics_to_save = {
        "raw_defect_dice": float(raw_result["defect_dice"]),
        "raw_defect_iou": float(raw_result["defect_iou"]),
        "raw_defect_image_recall": float(raw_result["defect_image_recall"]),
        "raw_normal_fpr": float(raw_result["normal_fpr"]),
        "raw_normal_count": int(raw_result["normal_count"]),
        "raw_normal_fp_count": int(raw_result["normal_fp_count"]),
        "raw_normal_fp_pixel_sum": float(raw_result["normal_fp_pixel_sum"]),
        "raw_normal_fp_pixel_mean": float(raw_result["normal_fp_pixel_mean"]),
        "raw_normal_fp_pixel_median": float(raw_result["normal_fp_pixel_median"]),
        "raw_normal_fp_pixel_p95": float(raw_result["normal_fp_pixel_p95"]),
        "raw_normal_largest_fp_area_mean": float(raw_result["normal_largest_fp_area_mean"]),
        "raw_normal_largest_fp_area_median": float(raw_result["normal_largest_fp_area_median"]),
        "raw_normal_largest_fp_area_p95": float(raw_result["normal_largest_fp_area_p95"]),
        "raw_normal_largest_fp_area_max": float(raw_result["normal_largest_fp_area_max"]),
        "raw_pixel_precision_defect_macro": float(raw_result["pixel_precision_defect_macro"]),
        "raw_pixel_recall_defect_macro": float(raw_result["pixel_recall_defect_macro"]),
        "raw_pixel_f1_defect_macro": float(raw_result["pixel_f1_defect_macro"]),
        "raw_pixel_precision_labeled_micro": float(raw_result["pixel_precision_labeled_micro"]),
        "raw_pixel_recall_labeled_micro": float(raw_result["pixel_recall_labeled_micro"]),
        "raw_pixel_f1_labeled_micro": float(raw_result["pixel_f1_labeled_micro"]),
        "raw_pixel_auprc_all_labeled": float(raw_result["pixel_auprc_all_labeled"]),
        "raw_component_recall_3px": float(raw_result["component_recall_3px"]),
        "raw_component_precision_3px": float(raw_result["component_precision_3px"]),
        "raw_component_f1_3px": float(raw_result["component_f1_3px"]),
        "raw_boundary_f1_3px": float(raw_result["boundary_f1_3px"]),
        "raw_threshold": float(raw_result["threshold"]),
        "raw_min_area": int(raw_result["min_area"]),
        "defect_dice": float(post_result["defect_dice"]),
        "defect_iou": float(post_result["defect_iou"]),
        "defect_image_recall": float(post_result["defect_image_recall"]),
        "normal_fpr": float(post_result["normal_fpr"]),
        "normal_fp_count": int(post_result["normal_fp_count"]),
        "normal_fp_pixel_sum": float(post_result["normal_fp_pixel_sum"]),
        "normal_fp_pixel_mean": float(post_result["normal_fp_pixel_mean"]),
        "normal_fp_pixel_median": float(post_result["normal_fp_pixel_median"]),
        "normal_fp_pixel_p95": float(post_result["normal_fp_pixel_p95"]),
        "normal_largest_fp_area_mean": float(post_result["normal_largest_fp_area_mean"]),
        "normal_largest_fp_area_median": float(post_result["normal_largest_fp_area_median"]),
        "normal_largest_fp_area_p95": float(post_result["normal_largest_fp_area_p95"]),
        "normal_largest_fp_area_max": float(post_result["normal_largest_fp_area_max"]),
        "pixel_precision_defect_macro": float(post_result["pixel_precision_defect_macro"]),
        "pixel_recall_defect_macro": float(post_result["pixel_recall_defect_macro"]),
        "pixel_f1_defect_macro": float(post_result["pixel_f1_defect_macro"]),
        "pixel_precision_labeled_micro": float(post_result["pixel_precision_labeled_micro"]),
        "pixel_recall_labeled_micro": float(post_result["pixel_recall_labeled_micro"]),
        "pixel_f1_labeled_micro": float(post_result["pixel_f1_labeled_micro"]),
        "pixel_auprc_all_labeled": float(post_result["pixel_auprc_all_labeled"]),
        "component_recall_3px": float(post_result["component_recall_3px"]),
        "component_precision_3px": float(post_result["component_precision_3px"]),
        "component_f1_3px": float(post_result["component_f1_3px"]),
        "boundary_f1_3px": float(post_result["boundary_f1_3px"]),
        "threshold": float(post_result["threshold"]),
        "min_area": int(post_result["min_area"]),
        "stage2_score": float(post_result["stage2_score"]),
        "defect_count": int(post_result["defect_count"]),
        "normal_count": int(post_result["normal_count"]),
        "target_normal_fpr": target_normal_fpr,
        "lambda_fpr_penalty": lambda_fpr_penalty,
        "checkpoint_best_threshold": checkpoint_best.get("threshold", None),
        "checkpoint_best_min_area": checkpoint_best.get("min_area", None),
    }

    save_json(save_dir / "val_metrics.json", metrics_to_save)

    per_image_fieldnames = [
        "image_name",
        "sample_type",
        "is_defect_image",
        "pred_has_positive",
        "dice",
        "iou",
        "threshold",
        "min_area",
        "fp_pixel_count",
        "largest_fp_component_area",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "tp_pixel_count",
        "fp_pixel_count_for_metric",
        "fn_pixel_count",
        "pred_pixel_count",
        "target_pixel_count",
        "component_recall_3px",
        "component_precision_3px",
        "component_f1_3px",
        "boundary_f1_3px",
    ]
    write_csv_rows(save_dir / "val_per_image.csv", post_result["per_image_rows"], per_image_fieldnames)

    search_fieldnames = [
        "threshold",
        "min_area",
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
        "pixel_precision_defect_macro",
        "pixel_recall_defect_macro",
        "pixel_f1_defect_macro",
        "pixel_precision_labeled_micro",
        "pixel_recall_labeled_micro",
        "pixel_f1_labeled_micro",
        "pixel_auprc_all_labeled",
        "component_recall_3px",
        "component_precision_3px",
        "component_f1_3px",
        "boundary_f1_3px",
        "stage2_score",
    ]
    write_csv_rows(save_dir / "val_postprocess_search.csv", post_result["search_rows"], search_fieldnames)

    return metrics_to_save


def main():
    args = parse_args()

    cfg = load_stage_config(resolve_path(args.config), "stage2")
    metrics_to_save = evaluate_and_save_stage2(cfg=cfg)
    print(metrics_to_save)


if __name__ == "__main__":
    main()
