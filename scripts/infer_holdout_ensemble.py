import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets import ROIDataset, build_stage2_eval_transform
from src.metrics import evaluate_prob_maps, logits_to_probs, probs_to_binary_mask
from src.model import build_model_from_config
from src.samples import holdout_samples, load_samples
from src.trainer import load_checkpoint, predict_on_loader
from src.utils import ensure_dir, load_yaml, read_json, save_json, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Run frozen OOF-postprocessed holdout inference with a fold ensemble")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--folds", type=str, default="0,1,2,3")
    parser.add_argument("--output-dir", type=str, default="")
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
        if item:
            folds.append(int(item))
    if len(folds) == 0:
        raise ValueError("No folds were provided")
    return folds


def apply_fold_overrides(cfg, fold):
    cfg = dict(cfg)
    cfg["fold"] = int(fold)
    for template_key, target_key in [
        ("save_dir_template", "save_dir"),
        ("prototype_bank_path_template", "prototype_bank_path"),
    ]:
        if template_key in cfg:
            cfg[target_key] = str(cfg[template_key]).format(fold=fold)
    return cfg


def safe_output_stem(value):
    text = str(value).strip() or "sample"
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_prob_map(path, probs):
    path = Path(path)
    ensure_dir(path.parent)
    image = (np.clip(np.asarray(probs, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def save_binary_mask(path, mask):
    path = Path(path)
    ensure_dir(path.parent)
    image = (np.asarray(mask).astype(np.uint8) > 0).astype(np.uint8) * 255
    Image.fromarray(image, mode="L").save(path)


def load_holdout_rows(cfg):
    return holdout_samples(load_samples(cfg["samples_path"], PROJECT_ROOT))


def load_postprocess_params(cfg):
    metrics = read_json(resolve_path(cfg["global_postprocess_path"]))
    return float(metrics["threshold"]), int(metrics.get("min_area", 0)), metrics


def quantitative_holdout_rows(rows):
    selected = []
    for row in rows:
        sample_type = str(row.get("sample_type", "")).strip()
        if sample_type == "defect" and str(row.get("mask_path", "")).strip() != "":
            selected.append(row)
        elif sample_type == "normal":
            selected.append(row)
    return selected


def predict_fold(cfg, fold, rows, device):
    fold_cfg = apply_fold_overrides(cfg, fold)
    dataset = ROIDataset(
        rows,
        image_size=int(fold_cfg.get("image_size", 640)),
        transform=build_stage2_eval_transform(int(fold_cfg.get("image_size", 640)), cfg=fold_cfg),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(fold_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(fold_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    model = build_model_from_config(fold_cfg)
    model.to(device)
    checkpoint_path = resolve_path(fold_cfg["save_dir"]) / "best_stage2.pt"
    load_checkpoint(checkpoint_path, model, map_location=device)
    predictions = predict_on_loader(
        model,
        loader,
        device,
        progress_desc=f"holdout fold {fold}",
        amp_enabled=bool(fold_cfg.get("amp", True)),
    )
    prob_maps = [logits_to_probs(item["logits"]).squeeze() for item in predictions]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prob_maps, predictions


def main():
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    folds = parse_folds(args.folds)
    rows = load_holdout_rows(cfg)
    output_dir = resolve_path(args.output_dir) if args.output_dir else resolve_path(cfg["global_postprocess_path"]).parent / "holdout_ensemble"
    ensure_dir(output_dir)

    threshold, min_area, postprocess_metrics = load_postprocess_params(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "auto")).lower() != "cpu" else "cpu")

    ensemble_sum = None
    reference_predictions = None
    for fold in folds:
        fold_probs, predictions = predict_fold(cfg, fold, rows, device)
        stacked = np.stack(fold_probs, axis=0)
        ensemble_sum = stacked if ensemble_sum is None else ensemble_sum + stacked
        if reference_predictions is None:
            reference_predictions = predictions

    ensemble_probs = ensemble_sum / float(len(folds))

    prob_dir = output_dir / "prob_maps"
    raw_mask_dir = output_dir / "raw_binary_masks"
    post_mask_dir = output_dir / "masks"
    summary_rows = []
    quantitative_indices = []

    quantitative_ids = {str(row.get("sample_id", "")) for row in quantitative_holdout_rows(rows)}
    for index, (row, item, prob_map) in enumerate(zip(rows, reference_predictions, ensemble_probs)):
        sample_id = str(row.get("sample_id") or item.get("sample_id") or "").strip()
        image_name = str(row.get("image_name") or item.get("image_name") or "")
        stem = safe_output_stem(sample_id or Path(image_name).stem)
        raw_binary_mask = probs_to_binary_mask(prob_map, threshold=threshold, min_area=0)
        post_binary_mask = probs_to_binary_mask(prob_map, threshold=threshold, min_area=min_area)

        save_prob_map(prob_dir / f"{stem}.png", prob_map)
        save_binary_mask(raw_mask_dir / f"{stem}.png", raw_binary_mask)
        save_binary_mask(post_mask_dir / f"{stem}.png", post_binary_mask)

        if sample_id in quantitative_ids:
            quantitative_indices.append(index)

        summary_rows.append(
            {
                "sample_id": sample_id,
                "image_name": image_name,
                "device": str(row.get("device") or ""),
                "defect_class": str(row.get("defect_class") or ""),
                "sample_type": str(row.get("sample_type") or ""),
                "source_split": str(row.get("source_split") or ""),
                "holdout_reason": str(row.get("holdout_reason") or ""),
                "is_labeled": str(row.get("is_labeled") or ""),
                "threshold": threshold,
                "min_area": min_area,
                "folds": ",".join(str(fold) for fold in folds),
                "max_prob": float(np.max(prob_map)),
                "raw_positive_pixels": int(raw_binary_mask.sum()),
                "post_positive_pixels": int(post_binary_mask.sum()),
            }
        )

    write_csv_rows(
        output_dir / "inference_summary.csv",
        summary_rows,
        [
            "sample_id",
            "image_name",
            "device",
            "defect_class",
            "sample_type",
            "source_split",
            "holdout_reason",
            "is_labeled",
            "threshold",
            "min_area",
            "folds",
            "max_prob",
            "raw_positive_pixels",
            "post_positive_pixels",
        ],
    )

    if len(quantitative_indices) > 0:
        prob_maps = [ensemble_probs[index] for index in quantitative_indices]
        gt_masks = [reference_predictions[index]["mask"].squeeze().numpy() for index in quantitative_indices]
        sample_types = [str(rows[index].get("sample_type", "")) for index in quantitative_indices]
        image_names = [str(rows[index].get("image_name", "")) for index in quantitative_indices]
        metrics = evaluate_prob_maps(
            prob_maps=prob_maps,
            gt_masks=gt_masks,
            sample_types=sample_types,
            image_names=image_names,
            threshold=threshold,
            min_area=min_area,
            include_auprc=True,
        )
        metrics_to_save = {
            key: value
            for key, value in metrics.items()
            if key not in {"per_image_rows", "pred_masks"}
        }
        metrics_to_save.update(
            {
                "folds": folds,
                "threshold_source": str(resolve_path(cfg["global_postprocess_path"])),
                "oof_threshold": threshold,
                "oof_min_area": min_area,
                "oof_defect_dice": postprocess_metrics.get("defect_dice"),
                "quantitative_count": len(quantitative_indices),
            }
        )
        save_json(output_dir / "holdout_metrics.json", metrics_to_save)
        write_csv_rows(
            output_dir / "holdout_per_image.csv",
            metrics["per_image_rows"],
            [
                "image_name",
                "sample_type",
                "is_defect_image",
                "pred_has_positive",
                "dice",
                "iou",
                "threshold",
                "min_area",
                "fp_pixel_count",
                "tp_pixel_count",
                "fp_pixel_count_for_metric",
                "fn_pixel_count",
                "pred_pixel_count",
                "target_pixel_count",
                "pixel_precision",
                "pixel_recall",
                "pixel_f1",
                "largest_fp_component_area",
                "component_recall_3px",
                "component_precision_3px",
                "component_f1_3px",
                "boundary_f1_3px",
            ],
        )

    print(
        {
            "output_dir": str(output_dir),
            "holdout_count": len(rows),
            "quantitative_count": len(quantitative_indices),
            "threshold": threshold,
            "min_area": min_area,
            "folds": folds,
        }
    )


if __name__ == "__main__":
    main()
