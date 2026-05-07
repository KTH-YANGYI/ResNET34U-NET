import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATASET_DIR_NAME = "dataset0505_crop640_roi"
DEFAULT_DATASET_ROOT_CANDIDATES = [
    PROJECT_ROOT.parent / DATASET_DIR_NAME,
    PROJECT_ROOT / DATASET_DIR_NAME,
]
DEFAULT_MASK_DIR = PROJECT_ROOT / "generated_masks"
DEFAULT_SAMPLES_PATH = PROJECT_ROOT / "manifests" / "samples.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "manifests" / "samples_summary.json"


SAMPLE_FIELDNAMES = [
    "sample_id",
    "image_name",
    "image_path",
    "mask_path",
    "json_path",
    "sample_type",
    "is_labeled",
    "source_split",
    "device",
    "defect_class",
    "cv_fold",
    "split",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare lightweight sample manifest for ROI dataset")
    parser.add_argument("--dataset-root", type=str, default=str(default_dataset_root()), help="ROI dataset root")
    parser.add_argument("--samples-path", type=str, default=str(DEFAULT_SAMPLES_PATH), help="Output samples.csv path")
    parser.add_argument("--mask-dir", type=str, default=str(DEFAULT_MASK_DIR), help="Generated mask output directory")
    parser.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_PATH), help="Output summary JSON path")
    parser.add_argument("--n-folds", type=int, default=4, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def default_dataset_root():
    for candidate in DEFAULT_DATASET_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate

    return DEFAULT_DATASET_ROOT_CANDIDATES[0]


def resolve_project_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def path_to_str(path):
    return str(Path(path).resolve())


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_csv_rows(path, rows, fieldnames):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, obj):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def build_sample_id(device, defect_class, image_path):
    return f"{device}_{defect_class}_{Path(image_path).stem}"


def polygon_mask_from_labelme(json_path, fallback_image_path):
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    image_width = int(data.get("imageWidth") or 0)
    image_height = int(data.get("imageHeight") or 0)

    if image_width <= 0 or image_height <= 0:
        with Image.open(fallback_image_path) as image:
            image_width, image_height = image.size

    mask = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)

    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        polygon = [(float(x), float(y)) for x, y in points]
        draw.polygon(polygon, fill=255)

    return mask


def save_labelme_mask(json_path, image_path, output_path):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    mask = polygon_mask_from_labelme(json_path, image_path)
    mask.save(output_path)
    mask_np = np.array(mask)
    return int((mask_np > 0).sum())


def assign_cv_folds(rows, n_folds, seed):
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] != "trainval":
            row["cv_fold"] = ""
            continue
        key = (row["device"], row["sample_type"], row["defect_class"])
        grouped[key].append(row)

    rng = random.Random(seed)
    for key in sorted(grouped.keys()):
        key_rows = grouped[key]
        rng.shuffle(key_rows)
        for index, row in enumerate(key_rows):
            row["cv_fold"] = int(index % n_folds)


def scan_roi_dataset(dataset_root, mask_dir):
    dataset_root = Path(dataset_root)
    mask_dir = Path(mask_dir)

    rows = []
    mask_area_by_sample = {}

    for device_dir in sorted(dataset_root.iterdir()):
        if not device_dir.is_dir():
            continue
        device = device_dir.name
        if device.startswith("."):
            continue

        for class_dir in sorted(device_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            defect_class = class_dir.name
            if defect_class.startswith("."):
                continue

            for image_path in sorted(class_dir.glob("*.png")):
                json_path = image_path.with_suffix(".json")
                sample_id = build_sample_id(device, defect_class, image_path)
                mask_path = ""
                is_labeled = False
                sample_type = "normal" if defect_class == "normal" else "defect"
                split = "trainval"

                if defect_class == "crack":
                    if not json_path.exists():
                        raise FileNotFoundError(f"Missing LabelMe JSON for crack image: {image_path}")
                    output_mask_path = mask_dir / device / defect_class / f"{image_path.stem}.png"
                    mask_area_by_sample[sample_id] = save_labelme_mask(json_path, image_path, output_mask_path)
                    mask_path = path_to_str(output_mask_path)
                    is_labeled = True
                elif defect_class == "broken":
                    sample_type = "broken_unlabeled"
                    split = "holdout"

                rows.append(
                    {
                        "sample_id": sample_id,
                        "image_name": image_path.name,
                        "image_path": path_to_str(image_path),
                        "mask_path": mask_path,
                        "json_path": path_to_str(json_path) if json_path.exists() else "",
                        "sample_type": sample_type,
                        "is_labeled": str(bool(is_labeled)),
                        "source_split": split,
                        "device": device,
                        "defect_class": defect_class,
                        "cv_fold": "",
                        "split": split,
                    }
                )

    return rows, mask_area_by_sample


def count_by(rows, *keys):
    counter = Counter(tuple(str(row.get(key, "")).strip() for key in keys) for row in rows)
    return {"|".join(key): int(value) for key, value in sorted(counter.items())}


def build_summary(rows, mask_area_by_sample, n_folds, seed, dataset_root, samples_path, mask_dir):
    fold_counts = Counter()
    for row in rows:
        if str(row.get("cv_fold", "")).strip() != "":
            fold_counts[str(row["cv_fold"])] += 1

    mask_areas = list(mask_area_by_sample.values())
    if len(mask_areas) > 0:
        mask_area_summary = {
            "min": int(min(mask_areas)),
            "median": float(np.median(mask_areas)),
            "mean": float(np.mean(mask_areas)),
            "max": int(max(mask_areas)),
        }
    else:
        mask_area_summary = {"min": 0, "median": 0.0, "mean": 0.0, "max": 0}

    return {
        "dataset_root": path_to_str(dataset_root),
        "samples_path": path_to_str(samples_path),
        "mask_dir": path_to_str(mask_dir),
        "n_folds": int(n_folds),
        "seed": int(seed),
        "total_count": len(rows),
        "count_by_split": count_by(rows, "split"),
        "count_by_device_class": count_by(rows, "device", "defect_class"),
        "count_by_sample_type": count_by(rows, "sample_type"),
        "fold_counts": dict(sorted(fold_counts.items(), key=lambda item: item[0])),
        "mask_area_px": mask_area_summary,
    }


def main():
    args = parse_args()
    dataset_root = resolve_project_path(args.dataset_root)
    samples_path = resolve_project_path(args.samples_path)
    mask_dir = resolve_project_path(args.mask_dir)
    summary_path = resolve_project_path(args.summary_path)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    rows, mask_area_by_sample = scan_roi_dataset(dataset_root, mask_dir)
    assign_cv_folds(rows, n_folds=args.n_folds, seed=args.seed)
    rows = sorted(rows, key=lambda row: (row["split"], row["device"], row["defect_class"], row["image_name"]))

    write_csv_rows(samples_path, rows, SAMPLE_FIELDNAMES)
    summary = build_summary(rows, mask_area_by_sample, args.n_folds, args.seed, dataset_root, samples_path, mask_dir)
    save_json(summary_path, summary)

    print(
        {
            "samples_path": path_to_str(samples_path),
            "summary_path": path_to_str(summary_path),
            "total_count": len(rows),
            "trainval_count": sum(1 for row in rows if row["split"] == "trainval"),
            "holdout_count": sum(1 for row in rows if row["split"] == "holdout"),
        }
    )


if __name__ == "__main__":
    main()
