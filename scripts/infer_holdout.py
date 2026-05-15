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


from src.datasets import (
    ROIDataset,
    build_same_size_batch_sampler,
    build_stage2_eval_transform,
    resolve_stage2_image_size,
    stage2_batch_size_by_image_size,
    stage2_uses_native_size,
)
from src.metrics import evaluate_prob_maps, logits_to_probs, probs_to_binary_mask
from src.model import build_model_from_config
from src.samples import holdout_samples, load_samples
from src.trainer import load_checkpoint, predict_on_loader
from src.utils import ensure_dir, load_stage_config, read_csv_rows, read_json, save_json, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Infer holdout images with best stage2 model")
    parser.add_argument("--config", type=str, default="configs/canonical_baseline.yaml", help="Path to config file")
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def save_prob_map(path, probs):
    path = Path(path)
    ensure_dir(path.parent)

    probs = np.asarray(probs, dtype=np.float32)
    probs = np.clip(probs, 0.0, 1.0)
    image = (probs * 255.0).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def save_binary_mask(path, mask):
    path = Path(path)
    ensure_dir(path.parent)

    mask = (np.asarray(mask).astype(np.uint8) > 0).astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(path)


def safe_output_stem(value):
    text = str(value).strip()
    if text == "":
        text = "sample"
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def to_bool(value):
    if torch.is_tensor(value):
        return bool(value.item())
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_holdout_rows(cfg):
    samples_path = str(cfg.get("samples_path", "")).strip()
    if samples_path != "":
        return holdout_samples(load_samples(samples_path, PROJECT_ROOT))

    return read_csv_rows(resolve_path("manifests/defect_holdout_unlabeled.csv"))


def eval_batch_size_by_image_size(cfg):
    mapping = cfg.get("eval_batch_size_by_image_size", cfg.get("stage2_eval_batch_size_by_image_size", None))
    if mapping is None:
        return stage2_batch_size_by_image_size(cfg)

    return {str(key).strip(): int(value) for key, value in dict(mapping).items()}


def load_postprocess_params(cfg, save_dir):
    metrics_path = save_dir / "val_metrics.json"
    metrics = read_json(metrics_path)
    return {
        "threshold": float(metrics["threshold"]),
        "min_area": int(metrics.get("min_area", 0)),
        "postprocess_source": "val",
        "postprocess_path": str(metrics_path),
    }


def serializable_metrics(result):
    output = {}
    for key, value in result.items():
        if key in {"per_image_rows", "pred_masks", "search_rows"}:
            continue
        if isinstance(value, (np.integer,)):
            output[key] = int(value)
        elif isinstance(value, (np.floating,)):
            output[key] = float(value)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            output[key] = value
    return output


def prefix_metrics(prefix, result):
    return {
        f"{prefix}_{key}": value
        for key, value in serializable_metrics(result).items()
    }


def build_labeled_records(predictions):
    records = []
    for item in predictions:
        if not to_bool(item.get("is_labeled", False)):
            continue

        records.append(
            {
                "prediction": item,
                "prob_map": logits_to_probs(item["logits"]).squeeze(),
                "gt_mask": item["mask"].squeeze().numpy(),
                "sample_type": str(item.get("sample_type") or ""),
                "image_name": str(item.get("image_name") or ""),
            }
        )
    return records


def evaluate_records(records, threshold, min_area, include_auprc=False):
    return evaluate_prob_maps(
        prob_maps=[record["prob_map"] for record in records],
        gt_masks=[record["gt_mask"] for record in records],
        sample_types=[record["sample_type"] for record in records],
        image_names=[record["image_name"] for record in records],
        threshold=threshold,
        min_area=min_area,
        include_auprc=include_auprc,
    )


def enrich_per_image_rows(per_image_rows, records, postprocess_params):
    enriched_rows = []
    for metric_row, record in zip(per_image_rows, records):
        item = record["prediction"]
        enriched_rows.append(
            {
                "sample_id": str(item.get("sample_id") or ""),
                "image_name": str(item.get("image_name") or ""),
                "device": str(item.get("device") or ""),
                "defect_class": str(item.get("defect_class") or ""),
                "sample_type": str(item.get("sample_type") or ""),
                "source_split": str(item.get("source_split") or ""),
                "holdout_reason": str(item.get("holdout_reason") or ""),
                "is_labeled": str(to_bool(item.get("is_labeled", False))),
                "postprocess_source": postprocess_params["postprocess_source"],
                "postprocess_path": postprocess_params["postprocess_path"],
                **metric_row,
            }
        )
    return enriched_rows


def build_group_metric_rows(records, threshold, min_area):
    group_rows = []
    for group_by in ["device", "sample_type", "defect_class"]:
        groups = {}
        for record in records:
            item = record["prediction"]
            group_value = str(item.get(group_by) or "").strip() or "unknown"
            groups.setdefault(group_value, []).append(record)

        for group_value, group_records in sorted(groups.items(), key=lambda pair: pair[0]):
            result = evaluate_records(
                group_records,
                threshold=threshold,
                min_area=min_area,
                include_auprc=False,
            )
            metrics = serializable_metrics(result)
            group_rows.append(
                {
                    "group_by": group_by,
                    "group": group_value,
                    "count": len(group_records),
                    "threshold": float(threshold),
                    "min_area": int(min_area),
                    "defect_dice": metrics.get("defect_dice", 0.0),
                    "defect_iou": metrics.get("defect_iou", 0.0),
                    "defect_image_recall": metrics.get("defect_image_recall", 0.0),
                    "normal_fpr": metrics.get("normal_fpr", 0.0),
                    "normal_count": metrics.get("normal_count", 0),
                    "normal_fp_count": metrics.get("normal_fp_count", 0),
                    "pixel_precision_labeled_micro": metrics.get("pixel_precision_labeled_micro", 0.0),
                    "pixel_recall_labeled_micro": metrics.get("pixel_recall_labeled_micro", 0.0),
                    "pixel_f1_labeled_micro": metrics.get("pixel_f1_labeled_micro", 0.0),
                    "boundary_f1_3px": metrics.get("boundary_f1_3px", 0.0),
                }
            )

    return group_rows


def main():
    args = parse_args()

    cfg = load_stage_config(resolve_path(args.config), "stage2")

    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "auto")).lower() != "cpu" else "cpu")

    holdout_rows = load_holdout_rows(cfg)

    image_size = resolve_stage2_image_size(cfg)
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 0))

    dataset = ROIDataset(
        holdout_rows,
        image_size=image_size,
        transform=build_stage2_eval_transform(image_size, cfg=cfg),
    )

    if stage2_uses_native_size(cfg):
        batch_sampler = build_same_size_batch_sampler(
            rows=holdout_rows,
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
    checkpoint_path = save_dir / "best_stage2.pt"
    load_checkpoint(checkpoint_path, model, map_location=device)

    postprocess_params = load_postprocess_params(cfg, save_dir)
    threshold = float(postprocess_params["threshold"])
    min_area = int(postprocess_params["min_area"])
    include_auprc = bool(cfg.get("eval_include_auprc", False))

    predictions = predict_on_loader(model, loader, device)

    prob_dir = save_dir / "holdout" / "prob_maps"
    raw_mask_dir = save_dir / "holdout" / "raw_binary_masks"
    post_mask_dir = save_dir / "holdout" / "masks"
    ensure_dir(prob_dir)
    ensure_dir(raw_mask_dir)
    ensure_dir(post_mask_dir)

    summary_rows = []

    for item in predictions:
        sample_id = str(item.get("sample_id") or "").strip()
        image_name = str(item["image_name"])
        stem = safe_output_stem(sample_id or Path(image_name).stem)

        prob_map = logits_to_probs(item["logits"]).squeeze()
        raw_binary_mask = probs_to_binary_mask(prob_map, threshold=threshold, min_area=0)
        post_binary_mask = probs_to_binary_mask(prob_map, threshold=threshold, min_area=min_area)

        save_prob_map(prob_dir / f"{stem}.png", prob_map)
        save_binary_mask(raw_mask_dir / f"{stem}.png", raw_binary_mask)
        save_binary_mask(post_mask_dir / f"{stem}.png", post_binary_mask)

        summary_rows.append(
            {
                "sample_id": sample_id,
                "image_name": image_name,
                "device": str(item.get("device") or ""),
                "defect_class": str(item.get("defect_class") or ""),
                "sample_type": str(item.get("sample_type") or ""),
                "source_split": str(item.get("source_split") or ""),
                "holdout_reason": str(item.get("holdout_reason") or ""),
                "is_labeled": str(to_bool(item.get("is_labeled", False))),
                "threshold": threshold,
                "min_area": min_area,
                "postprocess_source": postprocess_params["postprocess_source"],
                "postprocess_path": postprocess_params["postprocess_path"],
                "max_prob": float(np.max(prob_map)),
                "raw_positive_pixels": int(raw_binary_mask.sum()),
                "post_positive_pixels": int(post_binary_mask.sum()),
            }
        )

    write_csv_rows(
        save_dir / "holdout" / "inference_summary.csv",
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
            "postprocess_source",
            "postprocess_path",
            "max_prob",
            "raw_positive_pixels",
            "post_positive_pixels",
        ],
    )

    labeled_records = build_labeled_records(predictions)
    metrics_available = len(labeled_records) > 0
    if metrics_available:
        raw_threshold = float(cfg.get("threshold", 0.5))
        raw_result = evaluate_records(
            labeled_records,
            threshold=raw_threshold,
            min_area=0,
            include_auprc=include_auprc,
        )
        post_result = evaluate_records(
            labeled_records,
            threshold=threshold,
            min_area=min_area,
            include_auprc=include_auprc,
        )

        holdout_metrics = {
            "metrics_available": True,
            "holdout_count": len(predictions),
            "labeled_count": len(labeled_records),
            "postprocess_source": postprocess_params["postprocess_source"],
            "postprocess_path": postprocess_params["postprocess_path"],
            **prefix_metrics("raw", raw_result),
            **serializable_metrics(post_result),
        }
        save_json(save_dir / "holdout" / "holdout_metrics.json", holdout_metrics)

        per_image_rows = enrich_per_image_rows(
            post_result["per_image_rows"],
            labeled_records,
            postprocess_params,
        )
        per_image_fieldnames = [
            "sample_id",
            "image_name",
            "device",
            "defect_class",
            "sample_type",
            "source_split",
            "holdout_reason",
            "is_labeled",
            "postprocess_source",
            "postprocess_path",
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
        write_csv_rows(save_dir / "holdout" / "holdout_per_image.csv", per_image_rows, per_image_fieldnames)

        group_rows = build_group_metric_rows(labeled_records, threshold=threshold, min_area=min_area)
        write_csv_rows(
            save_dir / "holdout" / "holdout_group_metrics.csv",
            group_rows,
            [
                "group_by",
                "group",
                "count",
                "threshold",
                "min_area",
                "defect_dice",
                "defect_iou",
                "defect_image_recall",
                "normal_fpr",
                "normal_count",
                "normal_fp_count",
                "pixel_precision_labeled_micro",
                "pixel_recall_labeled_micro",
                "pixel_f1_labeled_micro",
                "boundary_f1_3px",
            ],
        )
    else:
        save_json(
            save_dir / "holdout" / "holdout_metrics.json",
            {
                "metrics_available": False,
                "holdout_count": len(predictions),
                "labeled_count": 0,
                "reason": "No labeled holdout rows were found.",
                "postprocess_source": postprocess_params["postprocess_source"],
                "postprocess_path": postprocess_params["postprocess_path"],
            },
        )

    print(
        {
            "holdout_count": len(predictions),
            "labeled_count": len(labeled_records),
            "metrics_available": metrics_available,
            "threshold": threshold,
            "min_area": min_area,
            "postprocess_source": postprocess_params["postprocess_source"],
            "postprocess_path": postprocess_params["postprocess_path"],
        }
    )


if __name__ == "__main__":
    main()
