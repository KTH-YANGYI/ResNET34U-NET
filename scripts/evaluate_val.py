import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets import ROIDataset, build_stage2_eval_transform
from src.model import build_model
from src.trainer import load_checkpoint, validate_stage2
from src.utils import load_yaml, read_csv_rows, save_json, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate best stage2 model on validation set")
    parser.add_argument("--config", type=str, default="configs/stage2.yaml", help="Path to config file")
    parser.add_argument("--fold", type=int, default=None, help="Override fold in config")
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def apply_fold_overrides(cfg, fold):
    cfg = dict(cfg)
    cfg["fold"] = int(fold)

    template_keys = [
        ("defect_val_manifest_template", "defect_val_manifest"),
        ("normal_val_manifest_template", "normal_val_manifest"),
        ("save_dir_template", "save_dir"),
    ]

    for template_key, target_key in template_keys:
        if template_key in cfg:
            cfg[target_key] = str(cfg[template_key]).format(fold=fold)

    return cfg


def build_threshold_grid(cfg):
    if "threshold_grid" in cfg:
        return [float(item) for item in cfg["threshold_grid"]]

    start = float(cfg.get("threshold_grid_start", 0.10))
    end = float(cfg.get("threshold_grid_end", 0.90))
    step = float(cfg.get("threshold_grid_step", 0.02))

    values = []
    current = start

    while current <= end + 1e-8:
        values.append(round(current, 6))
        current += step

    return values


def build_min_area_grid(cfg):
    values = cfg.get("min_area_grid", [0, 8, 16, 24, 32, 48])
    return [int(item) for item in values]


def evaluate_and_save_stage2(cfg, fold, checkpoint_path=None, progress_desc=None):
    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "auto")).lower() != "cpu" else "cpu")

    defect_val_rows = read_csv_rows(resolve_path(cfg["defect_val_manifest"]))
    normal_val_rows = read_csv_rows(resolve_path(cfg["normal_val_manifest"]))
    rows = defect_val_rows + normal_val_rows

    image_size = int(cfg.get("image_size", 640))
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 0))

    dataset = ROIDataset(
        rows,
        image_size=image_size,
        transform=build_stage2_eval_transform(image_size, cfg=cfg),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(pretrained=False)
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

    raw_result = validate_stage2(
        model=model,
        loader=loader,
        device=device,
        threshold=raw_threshold,
        min_area=0,
        target_normal_fpr=target_normal_fpr,
        lambda_fpr_penalty=lambda_fpr_penalty,
        progress_desc=None if progress_desc is None else f"{progress_desc} raw",
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
    )

    checkpoint_best = checkpoint.get("best_stage2_result", {})

    metrics_to_save = {
        "fold": fold,
        "raw_defect_dice": float(raw_result["defect_dice"]),
        "raw_defect_iou": float(raw_result["defect_iou"]),
        "raw_defect_image_recall": float(raw_result["defect_image_recall"]),
        "raw_normal_fpr": float(raw_result["normal_fpr"]),
        "raw_threshold": float(raw_result["threshold"]),
        "raw_min_area": int(raw_result["min_area"]),
        "defect_dice": float(post_result["defect_dice"]),
        "defect_iou": float(post_result["defect_iou"]),
        "defect_image_recall": float(post_result["defect_image_recall"]),
        "normal_fpr": float(post_result["normal_fpr"]),
        "normal_fp_pixel_mean": float(post_result["normal_fp_pixel_mean"]),
        "normal_largest_fp_area_mean": float(post_result["normal_largest_fp_area_mean"]),
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
    ]
    write_csv_rows(save_dir / "val_per_image.csv", post_result["per_image_rows"], per_image_fieldnames)

    search_fieldnames = [
        "threshold",
        "min_area",
        "defect_dice",
        "defect_iou",
        "defect_image_recall",
        "normal_fpr",
        "normal_fp_pixel_mean",
        "normal_largest_fp_area_mean",
        "stage2_score",
    ]
    write_csv_rows(save_dir / "val_postprocess_search.csv", post_result["search_rows"], search_fieldnames)

    return metrics_to_save


def main():
    args = parse_args()

    cfg = load_yaml(resolve_path(args.config))
    fold = int(args.fold if args.fold is not None else cfg.get("fold", 0))
    cfg = apply_fold_overrides(cfg, fold)
    metrics_to_save = evaluate_and_save_stage2(cfg=cfg, fold=fold)
    print(metrics_to_save)


if __name__ == "__main__":
    main()
