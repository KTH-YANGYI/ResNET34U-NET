import argparse
import csv
import json
import random
import sys
import ast
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.samples import load_samples, split_samples_for_fold


MANIFEST_DIR = PROJECT_ROOT / "manifests"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


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


def load_yaml(path):
    data = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text == "" or text.startswith("#") or ":" not in text:
                continue

            key, raw_value = text.split(":", 1)
            key = key.strip()
            value = raw_value.strip()

            if value == "":
                data[key] = ""
                continue

            lower_value = value.lower()
            if lower_value in {"true", "false"}:
                data[key] = lower_value == "true"
                continue

            try:
                data[key] = ast.literal_eval(value)
                continue
            except Exception:
                pass

            try:
                if any(char in value for char in [".", "e", "E"]):
                    data[key] = float(value)
                else:
                    data[key] = int(value)
                continue
            except Exception:
                data[key] = value.strip("\"'")

    return data


def read_mask_binary(path):
    mask = Image.open(Path(path)).convert("L")
    return (np.array(mask) > 0).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description="Build stage1 patch indexes")
    parser.add_argument("--config", type=str, default="configs/stage1.yaml", help="Path to stage1 config")
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def normalize_int_list(value, default_values):
    if value is None:
        return [int(item) for item in default_values]

    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]

    return [int(value)]


def load_patch_cfg(config_path):
    cfg = load_yaml(resolve_path(config_path))

    return {
        "n_folds": int(cfg.get("n_folds", 4)),
        "samples_path": str(cfg.get("samples_path", "")).strip(),
        "patch_out_size": int(cfg.get("patch_out_size", 384)),
        "positive_center_crop_sizes": normalize_int_list(cfg.get("positive_center_crop_sizes"), [320, 384, 448]),
        "positive_shift_crop_sizes": normalize_int_list(cfg.get("positive_shift_crop_sizes"), [320, 384, 448]),
        "positive_context_crop_sizes": normalize_int_list(cfg.get("positive_context_crop_sizes"), [384, 448]),
        "positive_boundary_crop_sizes": normalize_int_list(cfg.get("positive_boundary_crop_sizes"), [320, 384, 448]),
        "negative_crop_sizes": normalize_int_list(cfg.get("negative_crop_sizes"), [320, 384, 448]),
        "positive_center_count_per_image": int(cfg.get("positive_center_count_per_image", 1)),
        "positive_shift_count_per_image": int(cfg.get("positive_shift_count_per_image", 2)),
        "positive_context_count_per_image": int(cfg.get("positive_context_count_per_image", 1)),
        "positive_boundary_count_per_image": int(cfg.get("positive_boundary_count_per_image", 2)),
        "near_miss_negative_count_per_image": int(cfg.get("near_miss_negative_count_per_image", 2)),
        "hard_negative_count_per_image": int(cfg.get("hard_negative_count_per_image", 3)),
        "hard_negative_safety_margin": int(cfg.get("hard_negative_safety_margin", 24)),
        "near_miss_margin_min": int(cfg.get("near_miss_margin_min", 8)),
        "near_miss_margin_max": int(cfg.get("near_miss_margin_max", 20)),
        "patch_dedup_iou": float(cfg.get("patch_dedup_iou", 0.70)),
        "max_components_per_image": int(cfg.get("max_components_per_image", 3)),
        "max_positive_patches_per_image": int(cfg.get("max_positive_patches_per_image", 6)),
        "max_attempts_per_patch": int(cfg.get("max_attempts_per_patch", 20)),
    }


def bbox_from_binary_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "bbox_w": x_max - x_min + 1,
        "bbox_h": y_max - y_min + 1,
        "center_x": (x_min + x_max) / 2.0,
        "center_y": (y_min + y_max) / 2.0,
    }


def mask_to_components(mask):
    binary_mask = (mask > 0).astype(np.uint8)

    if int(binary_mask.sum()) == 0:
        return []

    components = []
    height, width = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    neighbor_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    start_ys, start_xs = np.where(binary_mask > 0)

    for start_y, start_x in zip(start_ys, start_xs):
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        coords = []

        while len(stack) > 0:
            y, x = stack.pop()
            coords.append((y, x))

            for dy, dx in neighbor_offsets:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or binary_mask[ny, nx] == 0:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))

        if len(coords) == 0:
            continue

        ys = np.asarray([coord[0] for coord in coords], dtype=np.int32)
        xs = np.asarray([coord[1] for coord in coords], dtype=np.int32)
        area_px = int(len(coords))
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1

        component_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        component_mask[ys, xs] = 1

        components.append(
            {
                "area_px": area_px,
                "bbox": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "bbox_w": bbox_w,
                    "bbox_h": bbox_h,
                    "center_x": float(xs.mean()),
                    "center_y": float(ys.mean()),
                },
                "mask": component_mask,
            }
        )

    components.sort(key=lambda item: item["area_px"], reverse=True)

    for component_id, component in enumerate(components):
        component["component_id"] = component_id
        component["center_x"] = float(component["bbox"]["center_x"])
        component["center_y"] = float(component["bbox"]["center_y"])

    return components


def clip_crop_window(center_x, center_y, crop_size, image_w, image_h):
    crop_size = int(crop_size)

    if crop_size > image_w or crop_size > image_h:
        return None

    crop_x = int(round(center_x - crop_size / 2.0))
    crop_y = int(round(center_y - crop_size / 2.0))

    crop_x = max(0, min(crop_x, image_w - crop_size))
    crop_y = max(0, min(crop_y, image_h - crop_size))

    return crop_x, crop_y


def sample_crop_window_containing_box(box, crop_size, image_w, image_h, rng):
    crop_size = int(crop_size)

    x_low = max(0, int(box["x_max"]) - crop_size + 1)
    x_high = min(int(box["x_min"]), image_w - crop_size)
    y_low = max(0, int(box["y_max"]) - crop_size + 1)
    y_high = min(int(box["y_min"]), image_h - crop_size)

    if x_low > x_high or y_low > y_high:
        return None

    crop_x = rng.randint(int(x_low), int(x_high))
    crop_y = rng.randint(int(y_low), int(y_high))
    return crop_x, crop_y


def crop_window_to_box(crop_x, crop_y, crop_size):
    crop_x = int(crop_x)
    crop_y = int(crop_y)
    crop_size = int(crop_size)

    return {
        "x_min": crop_x,
        "y_min": crop_y,
        "x_max": crop_x + crop_size - 1,
        "y_max": crop_y + crop_size - 1,
    }


def box_area(box):
    width = max(0, int(box["x_max"]) - int(box["x_min"]) + 1)
    height = max(0, int(box["y_max"]) - int(box["y_min"]) + 1)
    return width * height


def box_iou(box_a, box_b):
    inter_x_min = max(int(box_a["x_min"]), int(box_b["x_min"]))
    inter_y_min = max(int(box_a["y_min"]), int(box_b["y_min"]))
    inter_x_max = min(int(box_a["x_max"]), int(box_b["x_max"]))
    inter_y_max = min(int(box_a["y_max"]), int(box_b["y_max"]))

    if inter_x_min > inter_x_max or inter_y_min > inter_y_max:
        return 0.0

    intersection = (inter_x_max - inter_x_min + 1) * (inter_y_max - inter_y_min + 1)
    union = box_area(box_a) + box_area(box_b) - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def boxes_intersect(box_a, box_b):
    if int(box_a["x_max"]) < int(box_b["x_min"]):
        return False
    if int(box_a["x_min"]) > int(box_b["x_max"]):
        return False
    if int(box_a["y_max"]) < int(box_b["y_min"]):
        return False
    if int(box_a["y_min"]) > int(box_b["y_max"]):
        return False
    return True


def expand_bbox(box, margin, image_w, image_h):
    return {
        "x_min": max(0, int(box["x_min"]) - int(margin)),
        "y_min": max(0, int(box["y_min"]) - int(margin)),
        "x_max": min(int(image_w) - 1, int(box["x_max"]) + int(margin)),
        "y_max": min(int(image_h) - 1, int(box["y_max"]) + int(margin)),
    }


def window_has_positive(mask, crop_x, crop_y, crop_size):
    patch_mask = mask[int(crop_y):int(crop_y) + int(crop_size), int(crop_x):int(crop_x) + int(crop_size)]
    return bool((patch_mask > 0).any())


def choose_valid_crop_size(crop_sizes, image_w, image_h, rng):
    valid_sizes = [int(item) for item in crop_sizes if int(item) <= image_w and int(item) <= image_h]
    if len(valid_sizes) == 0:
        return None
    return int(rng.choice(valid_sizes))


def patch_family_from_type(patch_type, is_replay=False):
    if is_replay:
        return "replay"

    if str(patch_type).startswith("positive"):
        return "positive"

    if patch_type in {"near_miss_negative", "hard_negative"}:
        return "defect_negative"

    if patch_type == "normal_negative":
        return "normal_negative"

    return "unknown"


def build_patch_row(
    row,
    patch_index,
    patch_type,
    crop_x,
    crop_y,
    crop_size,
    out_size,
    component_id="",
    component_area_px="",
    is_replay=False,
    replay_score=0.0,
    source_epoch="",
):
    patch_id = f"{row['sample_id']}_{patch_type}_{patch_index:02d}"

    return {
        "patch_id": patch_id,
        "base_sample_id": row["sample_id"],
        "image_path": row["image_path"],
        "mask_path": row["mask_path"],
        "patch_type": patch_type,
        "patch_family": patch_family_from_type(patch_type, is_replay=is_replay),
        "video_id": str(row.get("video_id", "")).strip(),
        "video_name": str(row.get("video_name", "")).strip(),
        "frame_id": str(row.get("frame_id", "")).strip(),
        "sample_type": str(row.get("sample_type", "")).strip(),
        "source_split": str(row.get("source_split", "")).strip(),
        "crop_x": int(crop_x),
        "crop_y": int(crop_y),
        "crop_size": int(crop_size),
        "out_size": int(out_size),
        "component_id": "" if component_id == "" else int(component_id),
        "component_area_px": "" if component_area_px == "" else int(component_area_px),
        "is_replay": int(bool(is_replay)),
        "replay_score": float(replay_score),
        "source_epoch": "" if source_epoch == "" else int(source_epoch),
    }


def add_patch_row_if_new(
    patch_rows,
    used_windows,
    row,
    patch_index,
    patch_type,
    crop_x,
    crop_y,
    crop_size,
    out_size,
    dedup_iou,
    component_id="",
    component_area_px="",
    is_replay=False,
    replay_score=0.0,
    source_epoch="",
):
    candidate_box = crop_window_to_box(crop_x, crop_y, crop_size)

    for used_window in used_windows:
        if box_iou(candidate_box, used_window["box"]) > float(dedup_iou):
            return False

    patch_row = build_patch_row(
        row=row,
        patch_index=patch_index,
        patch_type=patch_type,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_size=crop_size,
        out_size=out_size,
        component_id=component_id,
        component_area_px=component_area_px,
        is_replay=is_replay,
        replay_score=replay_score,
        source_epoch=source_epoch,
    )

    patch_rows.append(patch_row)
    used_windows.append({"box": candidate_box, "patch_type": patch_type})
    return True


def component_relative_position(component, crop_x, crop_y, crop_size):
    rel_x = (float(component["center_x"]) - float(crop_x)) / float(crop_size)
    rel_y = (float(component["center_y"]) - float(crop_y)) / float(crop_size)
    return rel_x, rel_y


def component_is_near_patch_edge(component, crop_x, crop_y, crop_size, edge_band_ratio=0.35):
    rel_x, rel_y = component_relative_position(component, crop_x, crop_y, crop_size)

    if rel_x < 0.0 or rel_x > 1.0 or rel_y < 0.0 or rel_y > 1.0:
        return False

    return (
        rel_x <= edge_band_ratio
        or rel_x >= (1.0 - edge_band_ratio)
        or rel_y <= edge_band_ratio
        or rel_y >= (1.0 - edge_band_ratio)
    )


def generate_positive_center_candidate(component, crop_size, image_w, image_h, rng):
    del rng
    return clip_crop_window(component["center_x"], component["center_y"], crop_size, image_w, image_h)


def generate_positive_shift_candidate(component, crop_size, image_w, image_h, rng):
    sampled_window = sample_crop_window_containing_box(component["bbox"], crop_size, image_w, image_h, rng)
    if sampled_window is not None:
        return sampled_window

    return clip_crop_window(component["center_x"], component["center_y"], crop_size, image_w, image_h)


def generate_positive_context_candidate(component, crop_size, image_w, image_h, rng):
    return generate_positive_shift_candidate(component, crop_size, image_w, image_h, rng)


def generate_positive_boundary_candidate(component, crop_size, image_w, image_h, rng):
    directions = ["left", "right", "top", "bottom"]
    rng.shuffle(directions)

    for direction in directions:
        edge_offset = rng.uniform(crop_size * 0.18, crop_size * 0.30)
        jitter = rng.uniform(-crop_size * 0.15, crop_size * 0.15)

        if direction == "left":
            center_x = component["center_x"] + crop_size / 2.0 - edge_offset
            center_y = component["center_y"] + jitter
        elif direction == "right":
            center_x = component["center_x"] - crop_size / 2.0 + edge_offset
            center_y = component["center_y"] + jitter
        elif direction == "top":
            center_x = component["center_x"] + jitter
            center_y = component["center_y"] + crop_size / 2.0 - edge_offset
        else:
            center_x = component["center_x"] + jitter
            center_y = component["center_y"] - crop_size / 2.0 + edge_offset

        candidate = clip_crop_window(center_x, center_y, crop_size, image_w, image_h)

        if candidate is None:
            continue

        crop_x, crop_y = candidate
        if component_is_near_patch_edge(component, crop_x, crop_y, crop_size):
            return crop_x, crop_y

    return None


def generate_near_miss_candidate(component, crop_size, image_w, image_h, rng, margin_min, margin_max):
    bbox = component["bbox"]
    directions = ["left", "right", "top", "bottom"]
    rng.shuffle(directions)

    for direction in directions:
        distance = rng.randint(max(1, int(margin_min)), max(1, int(margin_max)))
        jitter = rng.randint(-crop_size // 4, crop_size // 4)

        if direction == "left":
            crop_x = int(round(int(bbox["x_min"]) - distance - crop_size + 1))
            crop_y = int(round(component["center_y"] - crop_size / 2.0 + jitter))
        elif direction == "right":
            crop_x = int(round(int(bbox["x_max"]) + distance))
            crop_y = int(round(component["center_y"] - crop_size / 2.0 + jitter))
        elif direction == "top":
            crop_x = int(round(component["center_x"] - crop_size / 2.0 + jitter))
            crop_y = int(round(int(bbox["y_min"]) - distance - crop_size + 1))
        else:
            crop_x = int(round(component["center_x"] - crop_size / 2.0 + jitter))
            crop_y = int(round(int(bbox["y_max"]) + distance))

        crop_x = max(0, min(crop_x, image_w - crop_size))
        crop_y = max(0, min(crop_y, image_h - crop_size))

        return crop_x, crop_y

    return None


def append_positive_patches_for_image(row, mask, components, rng, patch_cfg, used_windows):
    if len(components) == 0:
        return []

    image_h, image_w = mask.shape[:2]
    out_size = int(patch_cfg["patch_out_size"])
    dedup_iou = float(patch_cfg["patch_dedup_iou"])
    patch_rows = []
    patch_index_by_type = Counter()

    generation_steps = [
        (
            "positive_boundary",
            int(patch_cfg["positive_boundary_count_per_image"]),
            patch_cfg["positive_boundary_crop_sizes"],
            generate_positive_boundary_candidate,
            True,
        ),
        (
            "positive_shift",
            int(patch_cfg["positive_shift_count_per_image"]),
            patch_cfg["positive_shift_crop_sizes"],
            generate_positive_shift_candidate,
            False,
        ),
        (
            "positive_center",
            int(patch_cfg["positive_center_count_per_image"]),
            patch_cfg["positive_center_crop_sizes"],
            generate_positive_center_candidate,
            False,
        ),
        (
            "positive_context",
            int(patch_cfg["positive_context_count_per_image"]),
            patch_cfg["positive_context_crop_sizes"],
            generate_positive_context_candidate,
            False,
        ),
    ]

    for patch_type, target_count, crop_sizes, generator_fn, require_boundary_position in generation_steps:
        for task_index in range(target_count):
            if len(patch_rows) >= int(patch_cfg["max_positive_patches_per_image"]):
                return patch_rows

            component = components[task_index % len(components)]
            max_attempts = int(patch_cfg["max_attempts_per_patch"])

            for _ in range(max_attempts):
                crop_size = choose_valid_crop_size(crop_sizes, image_w, image_h, rng)
                if crop_size is None:
                    break

                candidate = generator_fn(component, crop_size, image_w, image_h, rng)
                if candidate is None:
                    continue

                crop_x, crop_y = candidate

                if not window_has_positive(mask, crop_x, crop_y, crop_size):
                    continue

                if require_boundary_position and not component_is_near_patch_edge(component, crop_x, crop_y, crop_size):
                    continue

                added = add_patch_row_if_new(
                    patch_rows=patch_rows,
                    used_windows=used_windows,
                    row=row,
                    patch_index=patch_index_by_type[patch_type],
                    patch_type=patch_type,
                    crop_x=crop_x,
                    crop_y=crop_y,
                    crop_size=crop_size,
                    out_size=out_size,
                    dedup_iou=dedup_iou,
                    component_id=component["component_id"],
                    component_area_px=component["area_px"],
                )

                if added:
                    patch_index_by_type[patch_type] += 1
                    break

    return patch_rows


def append_near_miss_patches_for_image(row, mask, components, rng, patch_cfg, used_windows):
    if len(components) == 0:
        return []

    image_h, image_w = mask.shape[:2]
    out_size = int(patch_cfg["patch_out_size"])
    dedup_iou = float(patch_cfg["patch_dedup_iou"])
    patch_rows = []
    patch_index = 0

    for task_index in range(int(patch_cfg["near_miss_negative_count_per_image"])):
        component = components[task_index % len(components)]
        expanded_box = expand_bbox(
            component["bbox"],
            margin=int(patch_cfg["near_miss_margin_max"]),
            image_w=image_w,
            image_h=image_h,
        )

        for _ in range(int(patch_cfg["max_attempts_per_patch"])):
            crop_size = choose_valid_crop_size(patch_cfg["negative_crop_sizes"], image_w, image_h, rng)
            if crop_size is None:
                break

            candidate = generate_near_miss_candidate(
                component,
                crop_size,
                image_w,
                image_h,
                rng,
                margin_min=int(patch_cfg["near_miss_margin_min"]),
                margin_max=int(patch_cfg["near_miss_margin_max"]),
            )
            if candidate is None:
                continue

            crop_x, crop_y = candidate

            if window_has_positive(mask, crop_x, crop_y, crop_size):
                continue

            patch_box = crop_window_to_box(crop_x, crop_y, crop_size)
            if not boxes_intersect(patch_box, expanded_box):
                continue

            added = add_patch_row_if_new(
                patch_rows=patch_rows,
                used_windows=used_windows,
                row=row,
                patch_index=patch_index,
                patch_type="near_miss_negative",
                crop_x=crop_x,
                crop_y=crop_y,
                crop_size=crop_size,
                out_size=out_size,
                dedup_iou=dedup_iou,
                component_id=component["component_id"],
                component_area_px=component["area_px"],
            )

            if added:
                patch_index += 1
                break

    return patch_rows


def append_hard_negative_patches_for_image(row, mask, global_bbox, rng, patch_cfg, used_windows):
    if global_bbox is None:
        return []

    image_h, image_w = mask.shape[:2]
    out_size = int(patch_cfg["patch_out_size"])
    dedup_iou = float(patch_cfg["patch_dedup_iou"])
    forbidden_box = expand_bbox(
        global_bbox,
        margin=int(patch_cfg["hard_negative_safety_margin"]),
        image_w=image_w,
        image_h=image_h,
    )

    patch_rows = []
    patch_index = 0

    while patch_index < int(patch_cfg["hard_negative_count_per_image"]):
        attempts = 0
        added = False

        while attempts < int(patch_cfg["max_attempts_per_patch"]) * 3:
            attempts += 1

            crop_size = choose_valid_crop_size(patch_cfg["negative_crop_sizes"], image_w, image_h, rng)
            if crop_size is None:
                break

            crop_x = rng.randint(0, image_w - crop_size)
            crop_y = rng.randint(0, image_h - crop_size)

            if window_has_positive(mask, crop_x, crop_y, crop_size):
                continue

            patch_box = crop_window_to_box(crop_x, crop_y, crop_size)
            if boxes_intersect(patch_box, forbidden_box):
                continue

            added = add_patch_row_if_new(
                patch_rows=patch_rows,
                used_windows=used_windows,
                row=row,
                patch_index=patch_index,
                patch_type="hard_negative",
                crop_x=crop_x,
                crop_y=crop_y,
                crop_size=crop_size,
                out_size=out_size,
                dedup_iou=dedup_iou,
            )

            if added:
                patch_index += 1
                break

        if not added:
            break

    return patch_rows


def get_image_hw(image_path):
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        image_w, image_h = image.size
    return image_h, image_w


def make_normal_negative_patches(normal_rows, target_count, rng, patch_cfg):
    if target_count <= 0:
        return []

    normal_rows = list(normal_rows)
    if len(normal_rows) == 0:
        return []

    shuffled_rows = normal_rows[:]
    rng.shuffle(shuffled_rows)

    selected_rows = []

    for row in shuffled_rows:
        if len(selected_rows) >= target_count:
            break
        selected_rows.append(row)

    while len(selected_rows) < target_count:
        selected_rows.append(rng.choice(normal_rows))

    patch_rows = []
    used_windows_by_image = {}
    dedup_iou = float(patch_cfg["patch_dedup_iou"])
    out_size = int(patch_cfg["patch_out_size"])

    for patch_index, row in enumerate(selected_rows):
        image_path = row["image_path"]
        image_h, image_w = get_image_hw(image_path)

        if image_path not in used_windows_by_image:
            used_windows_by_image[image_path] = []

        used_windows = used_windows_by_image[image_path]

        for _ in range(int(patch_cfg["max_attempts_per_patch"])):
            crop_size = choose_valid_crop_size(patch_cfg["negative_crop_sizes"], image_w, image_h, rng)
            if crop_size is None:
                break

            crop_x = rng.randint(0, image_w - crop_size)
            crop_y = rng.randint(0, image_h - crop_size)

            added = add_patch_row_if_new(
                patch_rows=patch_rows,
                used_windows=used_windows,
                row=row,
                patch_index=patch_index,
                patch_type="normal_negative",
                crop_x=crop_x,
                crop_y=crop_y,
                crop_size=crop_size,
                out_size=out_size,
                dedup_iou=dedup_iou,
            )

            if added:
                break

    return patch_rows


PATCH_INDEX_FIELDNAMES = [
    "patch_id",
    "base_sample_id",
    "image_path",
    "mask_path",
    "patch_type",
    "patch_family",
    "video_id",
    "video_name",
    "frame_id",
    "sample_type",
    "source_split",
    "crop_x",
    "crop_y",
    "crop_size",
    "out_size",
    "component_id",
    "component_area_px",
    "is_replay",
    "replay_score",
    "source_epoch",
]


def sort_patch_rows(patch_rows):
    def sort_key(row):
        return (
            row.get("base_sample_id", ""),
            row.get("patch_family", ""),
            row.get("patch_type", ""),
            row.get("patch_id", ""),
        )

    return sorted(patch_rows, key=sort_key)


def count_patch_types(patch_rows):
    counter = Counter()

    for row in patch_rows:
        counter[str(row.get("patch_type", ""))] += 1

    return dict(sorted(counter.items(), key=lambda item: item[0]))


def count_patch_families(patch_rows):
    counter = Counter()

    for row in patch_rows:
        counter[str(row.get("patch_family", ""))] += 1

    return dict(sorted(counter.items(), key=lambda item: item[0]))


def build_patch_index_for_split(defect_rows, normal_rows, seed, patch_cfg):
    rng = random.Random(seed)

    positive_patch_rows = []
    near_miss_patch_rows = []
    hard_negative_patch_rows = []

    for row in defect_rows:
        mask = read_mask_binary(row["mask_path"])
        components = mask_to_components(mask)
        if len(components) == 0:
            continue

        components = components[: int(patch_cfg["max_components_per_image"])]
        used_windows = []

        positive_patch_rows.extend(
            append_positive_patches_for_image(
                row=row,
                mask=mask,
                components=components,
                rng=rng,
                patch_cfg=patch_cfg,
                used_windows=used_windows,
            )
        )

        near_miss_patch_rows.extend(
            append_near_miss_patches_for_image(
                row=row,
                mask=mask,
                components=components,
                rng=rng,
                patch_cfg=patch_cfg,
                used_windows=used_windows,
            )
        )

        hard_negative_patch_rows.extend(
            append_hard_negative_patches_for_image(
                row=row,
                mask=mask,
                global_bbox=bbox_from_binary_mask(mask),
                rng=rng,
                patch_cfg=patch_cfg,
                used_windows=used_windows,
            )
        )

    normal_negative_target = len(near_miss_patch_rows) + len(hard_negative_patch_rows)
    normal_negative_patch_rows = make_normal_negative_patches(
        normal_rows=normal_rows,
        target_count=normal_negative_target,
        rng=rng,
        patch_cfg=patch_cfg,
    )

    all_patch_rows = sort_patch_rows(
        positive_patch_rows + near_miss_patch_rows + hard_negative_patch_rows + normal_negative_patch_rows
    )

    summary = {
        "positive_count": len(positive_patch_rows),
        "near_miss_negative_count": len(near_miss_patch_rows),
        "hard_negative_count": len(hard_negative_patch_rows),
        "normal_negative_count": len(normal_negative_patch_rows),
        "total_count": len(all_patch_rows),
        "count_by_type": count_patch_types(all_patch_rows),
        "count_by_family": count_patch_families(all_patch_rows),
    }

    return all_patch_rows, summary


def write_patch_index_csv(path, patch_rows):
    path = Path(path)
    ensure_dir(path.parent)
    write_csv_rows(path, patch_rows, PATCH_INDEX_FIELDNAMES)


def main():
    args = parse_args()
    patch_cfg = load_patch_cfg(args.config)
    sample_rows = None
    if patch_cfg["samples_path"] != "":
        sample_rows = load_samples(patch_cfg["samples_path"], PROJECT_ROOT)

    summary = {
        "n_folds": int(patch_cfg["n_folds"]),
        "patch_out_size": int(patch_cfg["patch_out_size"]),
        "samples_path": patch_cfg["samples_path"],
        "patch_cfg": {
            key: value
            for key, value in patch_cfg.items()
            if key not in {"n_folds", "patch_out_size", "samples_path"}
        },
        "folds": [],
    }

    for fold_index in range(int(patch_cfg["n_folds"])):
        if sample_rows is not None:
            defect_train_rows, defect_val_rows, normal_train_rows, normal_val_rows = split_samples_for_fold(
                sample_rows,
                fold=fold_index,
            )
        else:
            defect_train_rows = read_csv_rows(MANIFEST_DIR / f"defect_fold{fold_index}_train.csv")
            defect_val_rows = read_csv_rows(MANIFEST_DIR / f"defect_fold{fold_index}_val.csv")
            normal_train_rows = read_csv_rows(MANIFEST_DIR / f"normal_fold{fold_index}_train.csv")
            normal_val_rows = read_csv_rows(MANIFEST_DIR / f"normal_fold{fold_index}_val.csv")

        train_seed = 1000 + fold_index
        val_seed = 2000 + fold_index

        train_patch_rows, train_summary = build_patch_index_for_split(
            defect_rows=defect_train_rows,
            normal_rows=normal_train_rows,
            seed=train_seed,
            patch_cfg=patch_cfg,
        )
        val_patch_rows, val_summary = build_patch_index_for_split(
            defect_rows=defect_val_rows,
            normal_rows=normal_val_rows,
            seed=val_seed,
            patch_cfg=patch_cfg,
        )

        train_index_path = MANIFEST_DIR / f"stage1_fold{fold_index}_train_index.csv"
        val_index_path = MANIFEST_DIR / f"stage1_fold{fold_index}_val_index.csv"

        write_patch_index_csv(train_index_path, train_patch_rows)
        write_patch_index_csv(val_index_path, val_patch_rows)

        fold_summary = {
            "fold_index": fold_index,
            "defect_train_image_count": len(defect_train_rows),
            "defect_val_image_count": len(defect_val_rows),
            "normal_train_image_count": len(normal_train_rows),
            "normal_val_image_count": len(normal_val_rows),
            "train_seed": train_seed,
            "val_seed": val_seed,
            "train_patch_summary": train_summary,
            "val_patch_summary": val_summary,
        }
        summary["folds"].append(fold_summary)

        print(
            f"fold {fold_index}: "
            f"train_total={train_summary['total_count']} "
            f"(pos={train_summary['positive_count']}, "
            f"near_miss={train_summary['near_miss_negative_count']}, "
            f"hard_neg={train_summary['hard_negative_count']}, "
            f"normal_neg={train_summary['normal_negative_count']}), "
            f"val_total={val_summary['total_count']} "
            f"(pos={val_summary['positive_count']}, "
            f"near_miss={val_summary['near_miss_negative_count']}, "
            f"hard_neg={val_summary['hard_negative_count']}, "
            f"normal_neg={val_summary['normal_negative_count']})"
        )

    save_json(MANIFEST_DIR / "stage1_patch_summary.json", summary)
    print("build_patch_index finished, stage1 patch indexes were written to manifests/.")


if __name__ == "__main__":
    main()
