import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATASET_DIR_NAME = "dataset_crack_normal_unet_811"
DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / DATASET_DIR_NAME
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
    "holdout_reason",
    "split",
    "video_id",
    "video_name",
    "frame_id",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare sample manifest for the split crack/normal dataset")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT), help="Split dataset root")
    parser.add_argument("--samples-path", type=str, default=str(DEFAULT_SAMPLES_PATH), help="Output samples.csv path")
    parser.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_PATH), help="Output summary JSON path")
    return parser.parse_args()


def resolve_project_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def path_to_str(path):
    return str(Path(path).resolve())


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def resolve_dataset_file(dataset_root, relative_value, absolute_value=""):
    relative_text = str(relative_value or "").strip()
    if relative_text != "":
        return dataset_root / relative_text

    absolute_text = str(absolute_value or "").strip()
    if absolute_text != "":
        return Path(absolute_text)

    return None


def build_sample_id(split, source_dataset, label, image_path):
    return f"{split}_{source_dataset}_{label}_{Path(image_path).stem}"


def load_split_manifest(dataset_root):
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Expected split dataset manifest: {manifest_path}")

    source_rows = read_csv_rows(manifest_path)
    if len(source_rows) == 0:
        raise ValueError(f"Dataset manifest is empty: {manifest_path}")

    required = {"split", "label", "image_relative_path", "mask_relative_path"}
    missing = required - set(source_rows[0].keys())
    if missing:
        raise ValueError(f"Dataset manifest is missing required column(s): {', '.join(sorted(missing))}")

    rows = []
    for source_row in source_rows:
        split = str(source_row.get("split", "")).strip()
        label = str(source_row.get("label", "")).strip()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split `{split}` in {manifest_path}")
        if label not in {"crack", "normal"}:
            raise ValueError(f"Unsupported label `{label}` in {manifest_path}")

        image_path = resolve_dataset_file(
            dataset_root,
            source_row.get("image_relative_path", ""),
            source_row.get("image_full_path", ""),
        )
        mask_path = resolve_dataset_file(
            dataset_root,
            source_row.get("mask_relative_path", ""),
            source_row.get("mask_full_path", ""),
        )
        json_path = resolve_dataset_file(
            dataset_root,
            source_row.get("annotation_relative_path", ""),
            source_row.get("annotation_full_path", ""),
        )
        if image_path is None or not image_path.exists():
            raise FileNotFoundError(f"Missing image listed in manifest: {image_path}")
        if mask_path is None or not mask_path.exists():
            raise FileNotFoundError(f"Missing mask listed in manifest: {mask_path}")

        source_dataset = str(source_row.get("source_dataset", "")).strip() or "unknown"
        video_name = str(source_row.get("source_video_name", "")).strip()
        video_id = Path(video_name).stem if video_name else ""

        rows.append(
            {
                "sample_id": build_sample_id(split, source_dataset, label, image_path),
                "image_name": image_path.name,
                "image_path": path_to_str(image_path),
                "mask_path": path_to_str(mask_path),
                "json_path": path_to_str(json_path) if json_path is not None and json_path.exists() else "",
                "sample_type": "defect" if label == "crack" else "normal",
                "is_labeled": str(label == "crack"),
                "source_split": split,
                "device": source_dataset,
                "defect_class": label,
                "holdout_reason": "test_split" if split == "test" else "",
                "split": split,
                "video_id": video_id,
                "video_name": video_name,
                "frame_id": str(source_row.get("source_frame", "")).strip(),
            }
        )

    split_order = {"train": 0, "val": 1, "test": 2}
    return sorted(
        rows,
        key=lambda row: (
            split_order[row["split"]],
            row["device"],
            row["defect_class"],
            row["image_name"],
        ),
    )


def count_by(rows, *keys):
    counter = Counter(tuple(str(row.get(key, "")).strip() for key in keys) for row in rows)
    return {"|".join(key): int(value) for key, value in sorted(counter.items())}


def build_summary(rows, dataset_root, samples_path):
    test_rows = [row for row in rows if row["split"] == "test"]
    return {
        "dataset_root": path_to_str(dataset_root),
        "samples_path": path_to_str(samples_path),
        "split_protocol": "explicit_train_val_test",
        "total_count": len(rows),
        "count_by_split": count_by(rows, "split"),
        "count_by_split_device_class": count_by(rows, "split", "device", "defect_class"),
        "count_by_device_class": count_by(rows, "device", "defect_class"),
        "count_by_sample_type": count_by(rows, "sample_type"),
        "holdout": {
            "source": "dataset_test_split",
            "count": len(test_rows),
            "count_by_device_class": count_by(test_rows, "device", "defect_class"),
        },
    }


def main():
    args = parse_args()
    dataset_root = resolve_project_path(args.dataset_root)
    samples_path = resolve_project_path(args.samples_path)
    summary_path = resolve_project_path(args.summary_path)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    rows = load_split_manifest(dataset_root)
    write_csv_rows(samples_path, rows, SAMPLE_FIELDNAMES)
    summary = build_summary(rows, dataset_root, samples_path)
    save_json(summary_path, summary)

    print(
        {
            "samples_path": path_to_str(samples_path),
            "summary_path": path_to_str(summary_path),
            "total_count": len(rows),
            "train_count": sum(1 for row in rows if row["split"] == "train"),
            "val_count": sum(1 for row in rows if row["split"] == "val"),
            "test_count": sum(1 for row in rows if row["split"] == "test"),
        }
    )


if __name__ == "__main__":
    main()
