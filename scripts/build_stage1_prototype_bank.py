import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets import PatchDataset, build_stage1_eval_transform
from src.model import build_model
from src.trainer import load_checkpoint
from src.utils import ensure_dir, load_yaml, read_csv_rows, save_json


POSITIVE_PATCH_TYPES = {
    "positive_boundary",
    "positive_shift",
    "positive_center",
    "positive_context",
}

NEGATIVE_PATCH_TYPES = {
    "near_miss_negative",
    "hard_negative",
    "normal_negative",
    "replay_defect_negative_fp",
    "replay_normal_negative_fp",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a Stage1 hard-negative prototype bank")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def apply_fold_overrides(cfg, fold):
    cfg = dict(cfg)
    cfg["fold"] = int(fold)
    for template_key, target_key in [
        ("stage1_checkpoint_template", "stage1_checkpoint"),
        ("prototype_bank_path_template", "prototype_bank_path"),
    ]:
        if template_key in cfg:
            cfg[target_key] = str(cfg[template_key]).format(fold=fold)
    return cfg


def is_positive_patch(row):
    patch_type = str(row.get("patch_type", "")).strip()
    patch_family = str(row.get("patch_family", "")).strip()
    if patch_type in POSITIVE_PATCH_TYPES:
        return True
    return patch_family == "positive"


def is_negative_patch(row):
    patch_type = str(row.get("patch_type", "")).strip()
    patch_family = str(row.get("patch_family", "")).strip()
    if patch_type in NEGATIVE_PATCH_TYPES:
        return True
    return patch_family == "negative"


def limit_rows(rows, max_count, seed):
    rows = list(rows)
    max_count = int(max_count)
    if max_count <= 0 or len(rows) <= max_count:
        return rows

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = torch.randperm(len(rows), generator=generator)[:max_count].tolist()
    indices = sorted(int(index) for index in indices)
    return [rows[index] for index in indices]


def count_patch_types(rows):
    counter = Counter()
    for row in rows:
        patch_type = str(row.get("patch_type", "")).strip() or "unknown"
        counter[patch_type] += 1
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def extract_features(model, rows, cfg, device):
    out_size = int(cfg.get("stage1_out_size", cfg.get("patch_out_size", 224)))
    batch_size = int(cfg.get("prototype_batch_size", cfg.get("batch_size", 64)))
    num_workers = int(cfg.get("num_workers", 4))

    dataset = PatchDataset(
        rows,
        transform=build_stage1_eval_transform(out_size, cfg=cfg),
        cache_enabled=bool(cfg.get("patch_worker_cache", False)),
        cache_max_items=int(cfg.get("patch_worker_cache_max_items", 0)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    captured = {}

    def hook(_module, _inputs, output):
        captured["features"] = output.detach()

    handle = model.center.register_forward_hook(hook)
    features = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                _ = model(images)
                batch_features = captured["features"]
                batch_features = batch_features.mean(dim=(2, 3))
                batch_features = F.normalize(batch_features, dim=1)
                features.append(batch_features.cpu())
    finally:
        handle.remove()

    if len(features) == 0:
        return torch.empty((0, 512), dtype=torch.float32)
    return torch.cat(features, dim=0).float()


def main():
    args = parse_args()
    cfg = apply_fold_overrides(load_yaml(resolve_path(args.config)), args.fold)
    seed = int(cfg.get("seed", 42)) + int(args.fold)

    manifest_path = resolve_path(f"manifests/stage1_fold{int(args.fold)}_train_index.csv")
    rows = read_csv_rows(manifest_path)
    all_pos_rows = [row for row in rows if is_positive_patch(row)]
    all_neg_rows = [row for row in rows if is_negative_patch(row)]
    pos_rows = limit_rows(
        all_pos_rows,
        int(cfg.get("prototype_pos_max", 128)),
        seed,
    )
    neg_rows = limit_rows(
        all_neg_rows,
        int(cfg.get("prototype_neg_max", 128)),
        seed + 1009,
    )
    if len(pos_rows) == 0 or len(neg_rows) == 0:
        raise ValueError(f"Need positive and negative prototypes, got pos={len(pos_rows)} neg={len(neg_rows)}")

    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "auto")).lower() != "cpu" else "cpu")
    model = build_model(pretrained=False)
    model.to(device)
    load_checkpoint(resolve_path(cfg["stage1_checkpoint"]), model, map_location=device, strict=True)

    pos_prototypes = extract_features(model, pos_rows, cfg, device)
    neg_prototypes = extract_features(model, neg_rows, cfg, device)

    if bool(cfg.get("prototype_l2_normalize", True)):
        pos_prototypes = F.normalize(pos_prototypes, dim=1)
        neg_prototypes = F.normalize(neg_prototypes, dim=1)

    output_path = resolve_path(cfg.get("prototype_bank_path", f"outputs/prototype_banks/fold{int(args.fold)}/prototype_bank.pt"))
    ensure_dir(output_path.parent)
    bank = {
        "pos_prototypes": pos_prototypes,
        "neg_prototypes": neg_prototypes,
        "meta": {
            "fold": int(args.fold),
            "stage1_checkpoint": str(resolve_path(cfg["stage1_checkpoint"])),
            "manifest_path": str(manifest_path),
            "num_pos_rows_before_limit": len(all_pos_rows),
            "num_neg_rows_before_limit": len(all_neg_rows),
            "pos_count": len(pos_rows),
            "neg_count": len(neg_rows),
            "pos_patch_type_counts_before_limit": count_patch_types(all_pos_rows),
            "neg_patch_type_counts_before_limit": count_patch_types(all_neg_rows),
            "pos_patch_type_counts": count_patch_types(pos_rows),
            "neg_patch_type_counts": count_patch_types(neg_rows),
            "feature_dim": int(pos_prototypes.shape[1]) if pos_prototypes.ndim == 2 else 0,
            "feature_layer": "center_gap",
            "normalize": bool(cfg.get("prototype_l2_normalize", True)),
        },
    }
    torch.save(bank, output_path)
    save_json(output_path.with_suffix(".json"), bank["meta"])
    print({"prototype_bank_path": str(output_path), **bank["meta"]})


if __name__ == "__main__":
    main()
