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
from src.metrics import logits_to_probs, probs_to_binary_mask
from src.model import build_model
from src.samples import holdout_samples, load_samples
from src.trainer import load_checkpoint, predict_on_loader
from src.utils import ensure_dir, load_yaml, read_csv_rows, read_json, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Infer holdout images with best stage2 model")
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

    if "save_dir_template" in cfg:
        cfg["save_dir"] = str(cfg["save_dir_template"]).format(fold=fold)

    return cfg


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


def main():
    args = parse_args()

    cfg = load_yaml(resolve_path(args.config))
    fold = int(args.fold if args.fold is not None else cfg.get("fold", 0))
    cfg = apply_fold_overrides(cfg, fold)

    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "auto")).lower() != "cpu" else "cpu")

    holdout_rows = load_holdout_rows(cfg)

    image_size = int(cfg.get("image_size", 640))
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 0))

    dataset = ROIDataset(
        holdout_rows,
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
    checkpoint_path = save_dir / "best_stage2.pt"
    load_checkpoint(checkpoint_path, model, map_location=device)

    val_metrics = read_json(save_dir / "val_metrics.json")
    threshold = float(val_metrics["threshold"])
    min_area = int(val_metrics.get("min_area", 0))

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
            "max_prob",
            "raw_positive_pixels",
            "post_positive_pixels",
        ],
    )

    print(
        {
            "fold": fold,
            "holdout_count": len(predictions),
            "threshold": threshold,
            "min_area": min_area,
        }
    )


if __name__ == "__main__":
    main()
