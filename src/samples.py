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


def _fold_value(row):
    text = str(row.get("cv_fold", "")).strip()
    if text == "":
        return None
    return int(text)


def _is_trainval(row):
    return str(row.get("split", "")).strip() == "trainval"


def is_labeled_defect(row):
    if not _is_trainval(row):
        return False
    if str(row.get("sample_type", "")).strip() != "defect":
        return False
    if not _to_bool(row.get("is_labeled", False)):
        return False
    return str(row.get("mask_path", "")).strip() != ""


def is_normal(row):
    if not _is_trainval(row):
        return False
    return str(row.get("sample_type", "")).strip() == "normal"


def split_samples_for_fold(rows, fold):
    fold = int(fold)
    defect_train_rows = []
    defect_val_rows = []
    normal_train_rows = []
    normal_val_rows = []

    for row in rows:
        row_fold = _fold_value(row)
        if row_fold is None:
            continue

        if is_labeled_defect(row):
            if row_fold == fold:
                defect_val_rows.append(row)
            else:
                defect_train_rows.append(row)
            continue

        if is_normal(row):
            if row_fold == fold:
                normal_val_rows.append(row)
            else:
                normal_train_rows.append(row)

    return defect_train_rows, defect_val_rows, normal_train_rows, normal_val_rows


def holdout_samples(rows):
    return [row for row in rows if str(row.get("split", "")).strip() == "holdout"]
