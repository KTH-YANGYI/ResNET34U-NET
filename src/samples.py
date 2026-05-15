import csv
from pathlib import Path


def resolve_project_path(path_text, project_root):
    path = Path(path_text)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_samples(samples_path, project_root):
    return read_csv_rows(resolve_project_path(samples_path, project_root))


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_defect(row):
    if str(row.get("split", "")).strip() not in {"train", "val"}:
        return False
    if str(row.get("sample_type", "")).strip() != "defect":
        return False
    if not _to_bool(row.get("is_labeled", False)):
        return False
    return str(row.get("mask_path", "")).strip() != ""


def _is_normal(row):
    if str(row.get("split", "")).strip() not in {"train", "val"}:
        return False
    return str(row.get("sample_type", "")).strip() == "normal"


def split_samples(rows):
    defect_train_rows = []
    defect_val_rows = []
    normal_train_rows = []
    normal_val_rows = []

    for row in rows:
        split = str(row.get("split", "")).strip()
        if split not in {"train", "val"}:
            continue

        if _is_defect(row):
            if split == "train":
                defect_train_rows.append(row)
            else:
                defect_val_rows.append(row)
            continue

        if _is_normal(row):
            if split == "train":
                normal_train_rows.append(row)
            else:
                normal_val_rows.append(row)

    return defect_train_rows, defect_val_rows, normal_train_rows, normal_val_rows


def holdout_samples(rows):
    return [row for row in rows if str(row.get("split", "")).strip() == "test"]
