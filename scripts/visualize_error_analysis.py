import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.evaluate_val import apply_fold_overrides, load_val_rows
from src.datasets import (
    ROIDataset,
    build_empty_mask,
    build_stage2_eval_transform,
    read_image_rgb,
    read_mask_binary,
    resize_image,
    resize_mask,
)
from src.metrics import dice_score, iou_score, largest_component_area, logits_to_probs, probs_to_binary_mask
from src.model import build_model
from src.trainer import load_checkpoint, predict_on_loader
from src.utils import ensure_dir, load_yaml, read_json, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Render validation error-analysis overlays for a Stage2 checkpoint")
    parser.add_argument("--config", type=str, default="configs/stage2.yaml", help="Path to stage2 config")
    parser.add_argument("--fold", type=int, default=None, help="Override fold in config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Defaults to best_stage2.pt")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory. Defaults to save_dir/error_analysis")
    parser.add_argument("--threshold", type=float, default=None, help="Override postprocess threshold")
    parser.add_argument("--min-area", type=int, default=None, help="Override postprocess min area")
    parser.add_argument("--max-items", type=int, default=24, help="Maximum overlays per error group")
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_device(cfg):
    device_text = str(cfg.get("device", "auto")).strip().lower()

    if device_text == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available() and device_text in {"auto", "cuda"}:
        return torch.device("cuda")

    if device_text == "cuda":
        raise ValueError("CUDA was requested but is not available")

    return torch.device(device_text)


def safe_stem(value):
    text = str(value).strip() or "sample"
    for char in ["/", "\\", " ", ":", "|"]:
        text = text.replace(char, "_")
    return text


def load_postprocess_params(cfg, save_dir, threshold_override=None, min_area_override=None):
    threshold = threshold_override
    min_area = min_area_override
    source = "cli"

    if threshold is None or min_area is None:
        metrics_path = save_dir / "val_metrics.json"
        if metrics_path.exists():
            metrics = read_json(metrics_path)
            if threshold is None:
                threshold = float(metrics["threshold"])
            if min_area is None:
                min_area = int(metrics.get("min_area", 0))
            source = str(metrics_path)

    if threshold is None:
        threshold = float(cfg.get("threshold", 0.5))
        source = "config"
    if min_area is None:
        min_area = 0
        source = "config" if source == "cli" else source

    return float(threshold), int(min_area), source


def is_defect_sample(row, gt_mask):
    sample_type = str(row.get("sample_type", "")).strip()
    if sample_type.startswith("normal"):
        return False
    if int(gt_mask.sum()) > 0:
        return True
    return not sample_type.startswith("normal")


def color_overlay(image, gt_mask, pred_mask, alpha=0.45):
    image = image.astype(np.float32)
    gt_mask = (gt_mask > 0)
    pred_mask = (pred_mask > 0)

    overlay = image.copy()
    gt_only = gt_mask & ~pred_mask
    pred_only = pred_mask & ~gt_mask
    overlap = gt_mask & pred_mask

    colors = {
        "gt_only": np.asarray([0.0, 255.0, 0.0], dtype=np.float32),
        "pred_only": np.asarray([255.0, 0.0, 0.0], dtype=np.float32),
        "overlap": np.asarray([255.0, 220.0, 0.0], dtype=np.float32),
    }

    overlay[gt_only] = (1.0 - alpha) * overlay[gt_only] + alpha * colors["gt_only"]
    overlay[pred_only] = (1.0 - alpha) * overlay[pred_only] + alpha * colors["pred_only"]
    overlay[overlap] = (1.0 - alpha) * overlay[overlap] + alpha * colors["overlap"]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def prob_heatmap(prob_map):
    prob = np.clip(np.asarray(prob_map, dtype=np.float32), 0.0, 1.0)
    heat = np.zeros((*prob.shape, 3), dtype=np.uint8)
    heat[..., 0] = (prob * 255.0).astype(np.uint8)
    heat[..., 1] = ((1.0 - np.abs(prob - 0.5) * 2.0) * 80.0).astype(np.uint8)
    heat[..., 2] = ((1.0 - prob) * 80.0).astype(np.uint8)
    return heat


def render_panel(image, gt_mask, pred_mask, prob_map, title):
    overlay = color_overlay(image, gt_mask, pred_mask)
    heat = prob_heatmap(prob_map)
    parts = [Image.fromarray(image), Image.fromarray(overlay), Image.fromarray(heat)]
    width = sum(part.width for part in parts)
    height = max(part.height for part in parts) + 34
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), title, fill=(0, 0, 0))

    x = 0
    for part in parts:
        canvas.paste(part, (x, 34))
        x += part.width
    return canvas


def make_contact_sheet(items, output_path, columns=3, thumb_width=360):
    if len(items) == 0:
        return

    thumbs = []
    for item in items:
        image = Image.open(item["overlay_path"]).convert("RGB")
        image = ImageOps.contain(image, (thumb_width, thumb_width))
        label_h = 32
        thumb = Image.new("RGB", (thumb_width, image.height + label_h), "white")
        draw = ImageDraw.Draw(thumb)
        draw.text((6, 8), item["label"], fill=(0, 0, 0))
        thumb.paste(image, ((thumb_width - image.width) // 2, label_h))
        thumbs.append(thumb)

    rows = int(np.ceil(len(thumbs) / columns))
    cell_h = max(thumb.height for thumb in thumbs)
    sheet = Image.new("RGB", (columns * thumb_width, rows * cell_h), "white")
    for index, thumb in enumerate(thumbs):
        row = index // columns
        col = index % columns
        sheet.paste(thumb, (col * thumb_width, row * cell_h))
    sheet.save(output_path)


def write_summary(path, rows):
    fieldnames = [
        "sample_id",
        "image_name",
        "sample_type",
        "is_defect_image",
        "pred_has_positive",
        "dice",
        "iou",
        "fp_pixel_count",
        "largest_fp_component_area",
        "max_prob",
        "threshold",
        "min_area",
        "category",
        "overlay_path",
    ]
    write_csv_rows(path, rows, fieldnames)


def main():
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    fold = int(args.fold if args.fold is not None else cfg.get("fold", 0))
    cfg = apply_fold_overrides(cfg, fold)

    save_dir = resolve_path(cfg["save_dir"])
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint is not None else save_dir / "best_stage2.pt"
    output_dir = resolve_path(args.output_dir) if args.output_dir is not None else save_dir / "error_analysis"
    overlay_dir = output_dir / "overlays"
    ensure_dir(overlay_dir)

    threshold, min_area, postprocess_source = load_postprocess_params(
        cfg,
        save_dir,
        threshold_override=args.threshold,
        min_area_override=args.min_area,
    )

    device = build_device(cfg)
    rows = load_val_rows(cfg, fold)
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

    model = build_model(
        pretrained=False,
        deep_supervision=bool(cfg.get("deep_supervision_enable", False)),
        boundary_aux=bool(cfg.get("boundary_aux_enable", False)),
    )
    model.to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)

    predictions = predict_on_loader(
        model,
        loader,
        device,
        progress_desc="Error analysis",
        amp_enabled=bool(cfg.get("amp", True)),
    )

    rows_by_sample_id = {str(row.get("sample_id", "")).strip(): row for row in rows}
    rows_by_image_name = {str(row.get("image_name", "")).strip(): row for row in rows}
    summary_rows = []
    render_candidates = []

    for item in predictions:
        sample_id = str(item.get("sample_id") or "").strip()
        image_name = str(item.get("image_name") or "").strip()
        row = rows_by_sample_id.get(sample_id) or rows_by_image_name.get(image_name)
        if row is None:
            continue

        image = resize_image(read_image_rgb(row["image_path"]), image_size)
        mask_path = str(row.get("mask_path", "")).strip()
        if mask_path:
            gt_mask = resize_mask(read_mask_binary(mask_path), image_size)
        else:
            gt_mask = build_empty_mask(image.shape[0], image.shape[1])

        prob_map = logits_to_probs(item["logits"]).squeeze()
        pred_mask = probs_to_binary_mask(prob_map, threshold=threshold, min_area=min_area)

        defect_image = is_defect_sample(row, gt_mask)
        pred_has_positive = int(pred_mask.sum()) > 0
        dice = dice_score(pred_mask, gt_mask) if defect_image else 0.0
        iou = iou_score(pred_mask, gt_mask) if defect_image else 0.0
        fp_pixel_count = int(pred_mask.sum()) if not defect_image else 0
        largest_fp_area = largest_component_area(pred_mask) if not defect_image else 0
        max_prob = float(np.max(prob_map))

        if defect_image and not pred_has_positive:
            category = "false_negative"
        elif defect_image:
            category = "defect"
        elif pred_has_positive:
            category = "false_positive"
        else:
            category = "true_negative"

        summary_row = {
            "sample_id": sample_id,
            "image_name": image_name,
            "sample_type": row.get("sample_type", ""),
            "is_defect_image": defect_image,
            "pred_has_positive": pred_has_positive,
            "dice": float(dice),
            "iou": float(iou),
            "fp_pixel_count": fp_pixel_count,
            "largest_fp_component_area": largest_fp_area,
            "max_prob": max_prob,
            "threshold": threshold,
            "min_area": min_area,
            "category": category,
            "overlay_path": "",
        }
        summary_rows.append(summary_row)
        render_candidates.append((summary_row, image, gt_mask, pred_mask, prob_map))

    false_positives = sorted(
        [item for item in render_candidates if item[0]["category"] == "false_positive"],
        key=lambda item: (item[0]["largest_fp_component_area"], item[0]["fp_pixel_count"], item[0]["max_prob"]),
        reverse=True,
    )
    false_negatives = sorted(
        [item for item in render_candidates if item[0]["category"] == "false_negative"],
        key=lambda item: item[0]["max_prob"],
    )
    worst_defects = sorted(
        [item for item in render_candidates if item[0]["is_defect_image"]],
        key=lambda item: (item[0]["dice"], -item[0]["max_prob"]),
    )

    groups = {
        "false_positives": false_positives[: args.max_items],
        "false_negatives": false_negatives[: args.max_items],
        "worst_defects": worst_defects[: args.max_items],
    }

    contact_items_by_group = {}
    for group_name, group_items in groups.items():
        contact_items = []
        for rank, (summary_row, image, gt_mask, pred_mask, prob_map) in enumerate(group_items, start=1):
            stem = safe_stem(summary_row["sample_id"] or Path(summary_row["image_name"]).stem)
            overlay_path = overlay_dir / f"{group_name}_{rank:03d}_{stem}.png"
            title = (
                f"{group_name} rank={rank} dice={summary_row['dice']:.3f} "
                f"fp={summary_row['fp_pixel_count']} maxp={summary_row['max_prob']:.3f}"
            )
            panel = render_panel(image, gt_mask, pred_mask, prob_map, title)
            panel.save(overlay_path)
            summary_row["overlay_path"] = str(overlay_path)
            contact_items.append(
                {
                    "overlay_path": overlay_path,
                    "label": f"{rank:02d} {summary_row['image_name']} d={summary_row['dice']:.2f}",
                }
            )
        contact_items_by_group[group_name] = contact_items

    write_summary(output_dir / "summary.csv", summary_rows)

    for group_name, contact_items in contact_items_by_group.items():
        make_contact_sheet(contact_items, output_dir / f"{group_name}_contact.png")

    with open(output_dir / "run_info.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "value"])
        writer.writeheader()
        writer.writerow({"key": "config", "value": args.config})
        writer.writerow({"key": "fold", "value": fold})
        writer.writerow({"key": "checkpoint", "value": str(checkpoint_path)})
        writer.writerow({"key": "threshold", "value": threshold})
        writer.writerow({"key": "min_area", "value": min_area})
        writer.writerow({"key": "postprocess_source", "value": postprocess_source})
        writer.writerow({"key": "output_dir", "value": str(output_dir)})

    print(
        {
            "fold": fold,
            "threshold": threshold,
            "min_area": min_area,
            "postprocess_source": postprocess_source,
            "summary_path": str(output_dir / "summary.csv"),
            "overlay_dir": str(overlay_dir),
        }
    )


if __name__ == "__main__":
    main()
