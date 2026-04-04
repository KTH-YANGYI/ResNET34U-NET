import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import PatchDataset, ROIDataset, build_stage1_eval_transform, build_stage2_eval_transform
from src.metrics import largest_component_area, logits_to_probs, probs_to_binary_mask
from src.trainer import predict_on_loader
from src.utils import save_json, seed_worker, write_csv_rows


STAGE1_PATCH_FIELDNAMES = [
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


STAGE2_HARD_NORMAL_FIELDNAMES = [
    "sample_id",
    "image_name",
    "image_path",
    "mask_path",
    "json_path",
    "sample_type",
    "is_labeled",
    "source_split",
    "video_id",
    "video_name",
    "frame_id",
    "is_hard_normal",
    "hard_score",
    "hard_fp_pixel_count",
    "hard_largest_fp_area",
    "hard_max_prob",
    "source_epoch",
]


def _to_int_or_none(value):
    try:
        text = str(value).strip()
        if text == "":
            return None
        return int(text)
    except Exception:
        return None


def _stable_row_key(row):
    return (
        str(row.get("sample_id", "")).strip(),
        str(row.get("patch_id", "")).strip(),
        str(row.get("image_name", "")).strip(),
        str(row.get("image_path", "")).strip(),
    )


def _frame_dedup_key(row, family_value):
    video_id = str(row.get("video_id", "")).strip()
    frame_id = _to_int_or_none(row.get("frame_id", ""))

    if video_id != "" and frame_id is not None:
        return video_id, frame_id, str(family_value).strip()

    return video_id, str(row.get("base_sample_id", row.get("sample_id", ""))).strip(), str(family_value).strip()


def _count_rows_by_field(rows, field_name):
    counter = Counter()
    for row in rows:
        counter[str(row.get(field_name, "")).strip()] += 1
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def _build_loader(dataset, batch_size, num_workers, device, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if int(num_workers) > 0 else None,
        generator=generator,
        persistent_workers=int(num_workers) > 0,
    )


def _stage1_positive_replay_score(prob_map, target_mask, threshold):
    pred_mask = (prob_map >= float(threshold)).astype(np.uint8)
    pred_sum = int(pred_mask.sum())
    target_sum = int(target_mask.sum())
    intersection = int((pred_mask * target_mask).sum())

    if pred_sum == 0 and target_sum == 0:
        dice = 1.0
    else:
        dice = float((2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6))

    score = 1.0 - dice
    if pred_sum == 0 and target_sum > 0:
        score += 1.0
    return float(score), pred_sum > 0


def _stage1_negative_replay_score(prob_map, threshold):
    pred_mask = (prob_map >= float(threshold)).astype(np.uint8)
    pred_sum = int(pred_mask.sum())
    if pred_sum <= 0:
        return 0.0, False

    max_prob = float(np.max(prob_map))
    pred_ratio = float(pred_sum / pred_mask.size)
    return pred_ratio + max_prob, True


def _select_top_rows(candidate_rows, target_count, dedup_family_value):
    if target_count <= 0:
        return []

    best_by_key = {}

    for item in candidate_rows:
        key = _frame_dedup_key(item["row"], dedup_family_value)
        existing = best_by_key.get(key)
        if existing is None or float(item["score"]) > float(existing["score"]):
            best_by_key[key] = item

    ranked_items = sorted(
        best_by_key.values(),
        key=lambda item: (float(item["score"]), _stable_row_key(item["row"])),
        reverse=True,
    )

    selected = []
    used_patch_ids = set()
    for item in ranked_items:
        patch_id = str(item["row"].get("patch_id", "")).strip()
        if patch_id in used_patch_ids:
            continue
        selected.append(item)
        used_patch_ids.add(patch_id)
        if len(selected) >= target_count:
            break

    return selected


def build_stage1_replay_rows(
    model,
    base_patch_rows,
    cfg,
    device,
    source_epoch,
):
    if len(base_patch_rows) == 0:
        return [], {"total_count": 0, "count_by_type": {}, "count_by_source_family": {}}

    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 0))
    threshold = float(cfg.get("stage1_eval_threshold", 0.5))
    seed = int(cfg.get("seed", 42)) + int(source_epoch) + 10000
    target_total = int(round(len(base_patch_rows) * float(cfg.get("stage1_replay_ratio", 0.0))))
    max_total = int(round(len(base_patch_rows) * float(cfg.get("stage1_max_replay_ratio", 0.20))))

    if max_total > 0:
        target_total = min(target_total, max_total)

    if target_total <= 0:
        return [], {"total_count": 0, "count_by_type": {}, "count_by_source_family": {}}

    dataset = PatchDataset(
        base_patch_rows,
        transform=build_stage1_eval_transform(int(cfg.get("patch_out_size", 384)), cfg=cfg),
    )
    loader = _build_loader(dataset, batch_size=batch_size, num_workers=num_workers, device=device, seed=seed)
    predictions = predict_on_loader(model, loader, device, progress_desc=f"Replay scan {source_epoch}")
    row_by_patch_id = {str(row.get("patch_id", "")).strip(): row for row in base_patch_rows}

    positive_candidates = []
    defect_negative_candidates = []
    normal_negative_candidates = []

    for item in predictions:
        patch_id = str(item.get("patch_id", "")).strip()
        row = row_by_patch_id.get(patch_id)
        if row is None:
            continue

        prob_map = logits_to_probs(item["logits"]).squeeze()
        target_mask = (item["mask"].squeeze().numpy() > 0).astype(np.uint8)
        base_family = str(row.get("patch_family", "")).strip() or "unknown"

        if base_family == "positive" or int(target_mask.sum()) > 0:
            score, pred_has_positive = _stage1_positive_replay_score(prob_map, target_mask, threshold)
            if score <= 0.0:
                continue
            positive_candidates.append(
                {
                    "row": row,
                    "score": score,
                    "patch_type": "replay_positive_fn" if not pred_has_positive else "replay_positive_hard",
                    "source_family": "positive",
                }
            )
            continue

        score, pred_has_positive = _stage1_negative_replay_score(prob_map, threshold)
        if not pred_has_positive or score <= 0.0:
            continue

        candidate = {
            "row": row,
            "score": score,
            "patch_type": "replay_defect_negative_fp" if base_family == "defect_negative" else "replay_normal_negative_fp",
            "source_family": base_family,
        }

        if base_family == "defect_negative":
            defect_negative_candidates.append(candidate)
        else:
            normal_negative_candidates.append(candidate)

    target_positive = int(math.ceil(target_total * 0.5))
    selected_positive = _select_top_rows(positive_candidates, target_positive, dedup_family_value="replay_positive")

    remaining = max(0, target_total - len(selected_positive))
    selected_defect_negative = _select_top_rows(defect_negative_candidates, remaining, dedup_family_value="replay_negative")
    remaining = max(0, remaining - len(selected_defect_negative))
    selected_normal_negative = _select_top_rows(normal_negative_candidates, remaining, dedup_family_value="replay_negative")

    selected_items = selected_positive + selected_defect_negative + selected_normal_negative

    replay_rows = []
    for item in selected_items:
        replay_row = dict(item["row"])
        base_patch_id = str(replay_row.get("patch_id", "")).strip()
        replay_row["is_replay"] = 1
        replay_row["replay_score"] = float(item["score"])
        replay_row["source_epoch"] = int(source_epoch)
        replay_row["patch_family"] = "replay"
        replay_row["patch_type"] = item["patch_type"]
        replay_row["patch_id"] = f"{base_patch_id}__replay_e{int(source_epoch):03d}"
        replay_rows.append(replay_row)

    replay_rows = sorted(
        replay_rows,
        key=lambda row: (
            str(row.get("video_id", "")).strip(),
            _to_int_or_none(row.get("frame_id", "")) if _to_int_or_none(row.get("frame_id", "")) is not None else -1,
            str(row.get("patch_type", "")).strip(),
            str(row.get("patch_id", "")).strip(),
        ),
    )

    summary = {
        "source_epoch": int(source_epoch),
        "total_count": len(replay_rows),
        "count_by_type": _count_rows_by_field(replay_rows, "patch_type"),
        "count_by_source_family": dict(
            sorted(Counter(item["source_family"] for item in selected_items).items(), key=lambda pair: pair[0])
        ),
    }
    return replay_rows, summary


def save_stage1_replay_outputs(save_dir, source_epoch, replay_rows, summary):
    save_dir = Path(save_dir)
    replay_dir = save_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)

    write_csv_rows(replay_dir / "replay_latest.csv", replay_rows, STAGE1_PATCH_FIELDNAMES)
    write_csv_rows(replay_dir / f"replay_epoch{int(source_epoch):03d}.csv", replay_rows, STAGE1_PATCH_FIELDNAMES)
    save_json(replay_dir / "replay_latest_summary.json", summary)


def row_sort_key_with_score(row, score_key):
    frame_value = _to_int_or_none(row.get("frame_id", ""))
    if frame_value is None:
        frame_value = -1
    return (
        float(row.get(score_key, 0.0)),
        str(row.get("video_id", "")).strip(),
        frame_value,
        str(row.get("sample_id", row.get("image_name", ""))).strip(),
    )


def sample_rows_with_frame_gap(rows, k, seed, min_frame_gap=0, score_key=None, descending=True):
    rows = list(rows)
    if k <= 0 or len(rows) == 0:
        return []

    rng = random.Random(seed)
    if score_key is None:
        ordered_rows = rows[:]
        rng.shuffle(ordered_rows)
    else:
        decorated_rows = []
        for row in rows:
            decorated_rows.append((float(row.get(score_key, 0.0)), rng.random(), row))
        ordered_rows = [row for _, _, row in sorted(decorated_rows, key=lambda item: (item[0], item[1]), reverse=descending)]

    selected = []
    selected_keys = set()
    frames_by_video = defaultdict(list)

    for row in ordered_rows:
        unique_key = _stable_row_key(row)
        if unique_key in selected_keys:
            continue

        video_id = str(row.get("video_id", "")).strip()
        frame_id = _to_int_or_none(row.get("frame_id", ""))
        allow_row = True

        if min_frame_gap > 0 and video_id != "" and frame_id is not None:
            existing_frames = frames_by_video[video_id]
            if any(abs(frame_id - existing_frame) <= int(min_frame_gap) for existing_frame in existing_frames):
                allow_row = False

        if not allow_row:
            continue

        selected.append(row)
        selected_keys.add(unique_key)
        if video_id != "" and frame_id is not None:
            frames_by_video[video_id].append(frame_id)

        if len(selected) >= k:
            break

    if len(selected) < k:
        for row in ordered_rows:
            unique_key = _stable_row_key(row)
            if unique_key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(unique_key)
            if len(selected) >= k:
                break

    return selected


def build_hard_normal_pool(
    model,
    normal_rows,
    cfg,
    device,
    source_epoch,
    threshold,
    min_area,
    defect_count,
    random_normal_count,
):
    if len(normal_rows) == 0:
        return [], {"pool_size": 0, "count_by_video": {}, "top_score": 0.0}

    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 0))
    seed = int(cfg.get("seed", 42)) + int(source_epoch) + 20000
    image_size = int(cfg.get("image_size", 640))
    hard_ratio = float(cfg.get("stage2_hard_normal_ratio", 0.0))
    target_hard_count = resolve_hard_normal_count(defect_count, random_normal_count, cfg)
    pool_factor = float(cfg.get("hard_normal_pool_factor", 3.0))
    pool_limit = max(target_hard_count, int(math.ceil(max(1, target_hard_count) * pool_factor)))

    if hard_ratio <= 0.0 and target_hard_count <= 0:
        return [], {"pool_size": 0, "count_by_video": {}, "top_score": 0.0}

    dataset = ROIDataset(
        normal_rows,
        image_size=image_size,
        transform=build_stage2_eval_transform(image_size, cfg=cfg),
    )
    loader = _build_loader(dataset, batch_size=batch_size, num_workers=num_workers, device=device, seed=seed)
    predictions = predict_on_loader(model, loader, device, progress_desc=f"Hard normal scan {source_epoch}")
    row_by_sample_id = {str(row.get("sample_id", "")).strip(): row for row in normal_rows}

    scored_rows = []
    for item in predictions:
        sample_id = str(item.get("sample_id", "")).strip()
        row = row_by_sample_id.get(sample_id)
        if row is None:
            continue

        prob_map = logits_to_probs(item["logits"]).squeeze()
        pred_mask = probs_to_binary_mask(prob_map, threshold=threshold, min_area=min_area)
        fp_pixel_count = int(pred_mask.sum())
        largest_fp_area = int(largest_component_area(pred_mask))
        max_prob = float(np.max(prob_map))

        if fp_pixel_count <= 0:
            continue

        scored_row = dict(row)
        scored_row["is_hard_normal"] = 1
        scored_row["hard_score"] = float(largest_fp_area * 1000000.0 + fp_pixel_count + max_prob)
        scored_row["hard_fp_pixel_count"] = fp_pixel_count
        scored_row["hard_largest_fp_area"] = largest_fp_area
        scored_row["hard_max_prob"] = max_prob
        scored_row["source_epoch"] = int(source_epoch)
        scored_rows.append(scored_row)

    selected_rows = sample_rows_with_frame_gap(
        scored_rows,
        k=pool_limit,
        seed=seed,
        min_frame_gap=int(cfg.get("frame_min_gap", 0)),
        score_key="hard_score",
        descending=True,
    )

    summary = {
        "source_epoch": int(source_epoch),
        "pool_size": len(selected_rows),
        "count_by_video": _count_rows_by_field(selected_rows, "video_id"),
        "top_score": float(max((row["hard_score"] for row in selected_rows), default=0.0)),
    }
    return selected_rows, summary


def save_stage2_hard_normal_outputs(save_dir, source_epoch, hard_rows, summary):
    save_dir = Path(save_dir)
    hard_dir = save_dir / "hard_normal"
    hard_dir.mkdir(parents=True, exist_ok=True)

    write_csv_rows(hard_dir / "hard_normal_latest.csv", hard_rows, STAGE2_HARD_NORMAL_FIELDNAMES)
    write_csv_rows(hard_dir / f"hard_normal_epoch{int(source_epoch):03d}.csv", hard_rows, STAGE2_HARD_NORMAL_FIELDNAMES)
    save_json(hard_dir / "hard_normal_latest_summary.json", summary)


def resolve_hard_normal_count(defect_count, random_normal_count, cfg):
    if "hard_normal_k_factor" in cfg:
        return int(round(int(defect_count) * float(cfg.get("hard_normal_k_factor", 0.0))))

    ratio = float(cfg.get("stage2_hard_normal_ratio", 0.0))
    if ratio <= 0.0:
        return 0

    if ratio >= 1.0:
        return int(random_normal_count)

    return int(round(float(random_normal_count) * ratio / max(1e-6, 1.0 - ratio)))
