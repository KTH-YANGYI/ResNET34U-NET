import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets import ROIDataset, build_stage2_eval_transform
from src.model import build_model
from src.trainer import load_checkpoint, validate_stage2
from src.utils import load_yaml, read_csv_rows, save_json, write_csv_rows

from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate best stage2 model on validation set")
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

    template_keys = [
        ("defect_val_manifest_template", "defect_val_manifest"),
        ("normal_val_manifest_template", "normal_val_manifest"),
        ("save_dir_template", "save_dir"),
    ]

    for template_key, target_key in template_keys:
        if template_key in cfg:
            cfg[target_key] = str(cfg[template_key]).format(fold=fold)

    return cfg


def main():
    args = parse_args()

    cfg = load_yaml(resolve_path(args.config))
    fold = int(args.fold if args.fold is not None else cfg.get("fold", 0))
    cfg = apply_fold_overrides(cfg, fold)

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

    result = validate_stage2(
        model=model,
        loader=loader,
        device=device,
        threshold=float(cfg.get("threshold", 0.5)),
    )

    metrics_to_save = {
        "fold": fold,
        "defect_dice": float(result["defect_dice"]),
        "defect_iou": float(result["defect_iou"]),
        "defect_image_recall": float(result["defect_image_recall"]),
        "normal_fpr": float(result["normal_fpr"]),
        "threshold": float(result["threshold"]),
        "defect_count": int(result["defect_count"]),
        "normal_count": int(result["normal_count"]),
    }

    save_json(save_dir / "val_metrics.json", metrics_to_save)

    fieldnames = [
        "image_name",
        "sample_type",
        "is_defect_image",
        "pred_has_positive",
        "dice",
        "iou",
        "threshold",
    ]
    write_csv_rows(save_dir / "val_per_image.csv", result["per_image_rows"], fieldnames)

    print(metrics_to_save)


if __name__ == "__main__":
    main()
