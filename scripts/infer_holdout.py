import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets import ROIDataset, build_stage2_eval_transform
from src.metrics import logits_to_probs, probs_to_binary_mask
from src.model import build_model
from src.trainer import load_checkpoint, predict_on_loader
from src.utils import ensure_dir, load_yaml, read_csv_rows, read_json


def parse_args():
    parser = argparse.ArgumentParser(description="Infer holdout images with best stage2 model")
    parser.add_argument("--config", type=str, default="configs/stage2.yaml", help="配置文件路径")
    parser.add_argument("--fold", type=int, default=None, help="覆盖配置中的 fold")
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
    """
    把概率图保存成灰度 png。
    """

    path = Path(path)
    ensure_dir(path.parent)

    probs = np.asarray(probs, dtype=np.float32)
    probs = np.clip(probs, 0.0, 1.0)
    image = (probs * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), image)


def save_binary_mask(path, mask):
    """
    把二值 mask 保存成 png。
    """

    path = Path(path)
    ensure_dir(path.parent)

    mask = (np.asarray(mask).astype(np.uint8) > 0).astype(np.uint8) * 255
    cv2.imwrite(str(path), mask)


def main():
    args = parse_args()

    cfg = load_yaml(resolve_path(args.config))
    fold = int(args.fold if args.fold is not None else cfg.get("fold", 0))
    cfg = apply_fold_overrides(cfg, fold)

    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "auto")).lower() != "cpu" else "cpu")

    holdout_rows = read_csv_rows(resolve_path("manifests/defect_holdout_unlabeled.csv"))

    image_size = int(cfg.get("image_size", 640))
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 0))

    dataset = ROIDataset(
        holdout_rows,
        image_size=image_size,
        transform=build_stage2_eval_transform(image_size),
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

    predictions = predict_on_loader(model, loader, device)

    prob_dir = save_dir / "holdout" / "prob_maps"
    mask_dir = save_dir / "holdout" / "masks"
    ensure_dir(prob_dir)
    ensure_dir(mask_dir)

    for item in predictions:
        image_name = str(item["image_name"])
        stem = Path(image_name).stem

        prob_map = logits_to_probs(item["logits"]).squeeze()
        binary_mask = probs_to_binary_mask(prob_map, threshold)

        save_prob_map(prob_dir / f"{stem}.png", prob_map)
        save_binary_mask(mask_dir / f"{stem}.png", binary_mask)

    print(
        {
            "fold": fold,
            "holdout_count": len(predictions),
            "threshold": threshold,
        }
    )


if __name__ == "__main__":
    main()
