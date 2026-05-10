import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets import ROIDataset, build_stage2_eval_transform
from src.metrics import logits_to_probs, search_postprocess_params
from src.model import build_model_from_config
from src.trainer import load_checkpoint, predict_on_loader
from src.utils import ensure_dir, load_yaml, save_json, write_csv_rows
from scripts.evaluate_val import build_min_area_grid, build_threshold_grid, load_val_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Search one global postprocess setting over pooled OOF validation predictions")
    parser.add_argument("--config", type=str, default="configs/stage2.yaml", help="Path to stage2 config file")
    parser.add_argument("--folds", type=str, default="0,1,2,3", help="Comma-separated fold ids")
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_folds(folds_text):
    folds = []
    for item in str(folds_text).split(","):
        item = item.strip()
        if item == "":
            continue
        folds.append(int(item))
    if len(folds) == 0:
        raise ValueError("No folds were provided")
    return folds


def apply_fold_overrides(cfg, fold):
    cfg = dict(cfg)
    cfg["fold"] = int(fold)

    template_keys = [
        ("defect_val_manifest_template", "defect_val_manifest"),
        ("normal_val_manifest_template", "normal_val_manifest"),
        ("save_dir_template", "save_dir"),
        ("prototype_bank_path_template", "prototype_bank_path"),
    ]

    for template_key, target_key in template_keys:
        if template_key in cfg:
            cfg[target_key] = str(cfg[template_key]).format(fold=fold)

    return cfg


def build_device(cfg):
    device_text = str(cfg.get("device", "auto")).strip().lower()

    if device_text == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available() and device_text in {"auto", "cuda"}:
        return torch.device("cuda")

    if device_text == "cuda":
        raise ValueError("CUDA was requested but is not available")

    return torch.device(device_text)


def predict_fold(cfg, fold, device):
    fold_cfg = apply_fold_overrides(cfg, fold)
    rows = load_val_rows(fold_cfg, fold)

    image_size = int(fold_cfg.get("image_size", 640))
    batch_size = int(fold_cfg.get("batch_size", 4))
    num_workers = int(fold_cfg.get("num_workers", 0))

    dataset = ROIDataset(
        rows,
        image_size=image_size,
        transform=build_stage2_eval_transform(image_size, cfg=fold_cfg),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model_from_config(fold_cfg)
    model.to(device)

    save_dir = resolve_path(fold_cfg["save_dir"])
    checkpoint_path = save_dir / "best_stage2.pt"
    load_checkpoint(checkpoint_path, model, map_location=device)

    predictions = predict_on_loader(
        model,
        loader,
        device,
        progress_desc=f"OOF fold {fold}",
        amp_enabled=bool(fold_cfg.get("amp", True)),
    )

    prob_maps = []
    gt_masks = []
    sample_types = []
    image_names = []
    fold_ids = []

    for item in predictions:
        prob_maps.append(logits_to_probs(item["logits"]).squeeze())
        gt_masks.append(item["mask"].squeeze().numpy())
        sample_types.append(item.get("sample_type", ""))
        image_names.append(item.get("image_name", ""))
        fold_ids.append(int(fold))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "prob_maps": prob_maps,
        "gt_masks": gt_masks,
        "sample_types": sample_types,
        "image_names": image_names,
        "fold_ids": fold_ids,
        "checkpoint_path": str(checkpoint_path),
    }


def build_output_paths(cfg):
    json_path = resolve_path(cfg.get("global_postprocess_path", "outputs/stage2/oof_global_postprocess.json"))
    output_dir = json_path.parent
    return {
        "json": json_path,
        "search_csv": output_dir / "oof_global_postprocess_search.csv",
        "per_image_csv": output_dir / "oof_per_image.csv",
    }


def main():
    args = parse_args()

    cfg = load_yaml(resolve_path(args.config))
    folds = parse_folds(args.folds)
    device = build_device(cfg)

    all_prob_maps = []
    all_gt_masks = []
    all_sample_types = []
    all_image_names = []
    all_fold_ids = []
    checkpoints = {}

    for fold in folds:
        fold_output = predict_fold(cfg, fold, device)
        all_prob_maps.extend(fold_output["prob_maps"])
        all_gt_masks.extend(fold_output["gt_masks"])
        all_sample_types.extend(fold_output["sample_types"])
        all_image_names.extend(fold_output["image_names"])
        all_fold_ids.extend(fold_output["fold_ids"])
        checkpoints[str(fold)] = fold_output["checkpoint_path"]

    target_normal_fpr = float(cfg.get("target_normal_fpr", 0.10))
    lambda_fpr_penalty = float(cfg.get("lambda_fpr_penalty", 2.0))
    search_output = search_postprocess_params(
        prob_maps=all_prob_maps,
        gt_masks=all_gt_masks,
        sample_types=all_sample_types,
        image_names=all_image_names,
        threshold_values=build_threshold_grid(cfg),
        min_area_values=build_min_area_grid(cfg),
        target_normal_fpr=target_normal_fpr,
        lambda_fpr_penalty=lambda_fpr_penalty,
        include_auprc=True,
    )

    best_result = search_output["best_result"]
    output_paths = build_output_paths(cfg)
    ensure_dir(output_paths["json"].parent)

    metrics_to_save = {
        key: value
        for key, value in best_result.items()
        if key not in {"per_image_rows", "pred_masks", "search_rows"}
    }
    metrics_to_save.update(
        {
            "folds": folds,
            "checkpoint_paths": checkpoints,
            "target_normal_fpr": target_normal_fpr,
            "lambda_fpr_penalty": lambda_fpr_penalty,
            "config_path": args.config,
        }
    )
    save_json(output_paths["json"], metrics_to_save)

    per_image_rows = []
    for row, fold in zip(best_result["per_image_rows"], all_fold_ids):
        row = dict(row)
        row["fold"] = int(fold)
        per_image_rows.append(row)

    per_image_fieldnames = [
        "fold",
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
    write_csv_rows(output_paths["per_image_csv"], per_image_rows, per_image_fieldnames)

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
    write_csv_rows(output_paths["search_csv"], search_output["search_rows"], search_fieldnames)

    print(metrics_to_save)


if __name__ == "__main__":
    main()
