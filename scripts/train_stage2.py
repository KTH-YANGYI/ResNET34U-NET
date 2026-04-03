import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets import ROIDataset, build_stage2_eval_transform, build_stage2_train_transform
from src.losses import BCEDiceLoss
from src.model import build_model
from src.trainer import (
    EarlyStopper,
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate_stage2,
)
from src.utils import ensure_dir, load_yaml, read_csv_rows, seed_worker, set_seed, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Train stage2 full-image segmentation model")
    parser.add_argument("--config", type=str, default="configs/stage2.yaml", help="配置文件路径")
    parser.add_argument("--fold", type=int, default=None, help="覆盖配置中的 fold")
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_device(cfg):
    device_text = str(cfg.get("device", "auto")).strip().lower()

    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_text == "cuda" and not torch.cuda.is_available():
        raise ValueError("配置要求使用 CUDA，但当前环境没有可用 CUDA")

    return torch.device(device_text)


def apply_fold_overrides(cfg, fold):
    cfg = dict(cfg)
    cfg["fold"] = int(fold)

    template_keys = [
        ("defect_train_manifest_template", "defect_train_manifest"),
        ("defect_val_manifest_template", "defect_val_manifest"),
        ("normal_train_manifest_template", "normal_train_manifest"),
        ("normal_val_manifest_template", "normal_val_manifest"),
        ("stage1_checkpoint_template", "stage1_checkpoint"),
        ("save_dir_template", "save_dir"),
    ]

    for template_key, target_key in template_keys:
        if template_key in cfg:
            cfg[target_key] = str(cfg[template_key]).format(fold=fold)

    return cfg


def build_scaler(cfg, device):
    amp_enabled = bool(cfg.get("amp", True))
    use_amp = amp_enabled and device.type == "cuda"

    if not use_amp:
        return None

    return torch.amp.GradScaler(device="cuda", enabled=True)


def sample_normal_rows(normal_rows, k, seed):
    """
    从 normal_train 里采样一部分 normal 图。
    """

    normal_rows = list(normal_rows)
    if k <= 0 or len(normal_rows) == 0:
        return []

    rng = random.Random(seed)

    if k >= len(normal_rows):
        sampled_rows = normal_rows[:]
        rng.shuffle(sampled_rows)
        return sampled_rows

    return rng.sample(normal_rows, k)


def build_epoch_train_rows(defect_train_rows, normal_train_rows, epoch_seed):
    """
    构造当前 epoch 的 stage2 训练样本。

    规则：
    1. defect_train 全部保留
    2. 再从 normal_train 里采样与 defect 数量接近的一部分
    3. 最后把两者混合并打乱
    """

    defect_train_rows = list(defect_train_rows)
    normal_train_rows = list(normal_train_rows)

    sampled_normal_rows = sample_normal_rows(
        normal_rows=normal_train_rows,
        k=len(defect_train_rows),
        seed=epoch_seed,
    )

    epoch_rows = defect_train_rows + sampled_normal_rows

    rng = random.Random(epoch_seed)
    rng.shuffle(epoch_rows)

    return epoch_rows


def build_stage2_train_loader(rows, cfg, device):
    image_size = int(cfg.get("image_size", 640))
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 8))
    seed = int(cfg.get("seed", 42))

    dataset = ROIDataset(
        rows,
        image_size=image_size,
        transform=build_stage2_train_transform(image_size),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = device.type == "cuda"
    worker_init_fn = seed_worker if num_workers > 0 else None
    persistent_workers = num_workers > 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=persistent_workers,
    )

    return loader


def build_stage2_val_loader(defect_val_rows, normal_val_rows, cfg, device):
    image_size = int(cfg.get("image_size", 640))
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 8))
    seed = int(cfg.get("seed", 42))

    rows = list(defect_val_rows) + list(normal_val_rows)

    dataset = ROIDataset(
        rows,
        image_size=image_size,
        transform=build_stage2_eval_transform(image_size),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = device.type == "cuda"
    worker_init_fn = seed_worker if num_workers > 0 else None
    persistent_workers = num_workers > 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=persistent_workers,
    )

    return loader


def stage2_result_is_better(current_result, best_result):
    """
    按 stage2 的规则比较两个 checkpoint 结果谁更好。
    """

    if best_result is None:
        return True

    current_ok = float(current_result["normal_fpr"]) <= 0.10
    best_ok = float(best_result["normal_fpr"]) <= 0.10

    if current_ok and not best_ok:
        return True

    if current_ok and best_ok:
        if float(current_result["defect_dice"]) > float(best_result["defect_dice"]):
            return True

        if (
            float(current_result["defect_dice"]) == float(best_result["defect_dice"])
            and float(current_result["normal_fpr"]) < float(best_result["normal_fpr"])
        ):
            return True

        return False

    if not current_ok and not best_ok:
        if float(current_result["normal_fpr"]) < float(best_result["normal_fpr"]):
            return True

        if (
            float(current_result["normal_fpr"]) == float(best_result["normal_fpr"])
            and float(current_result["defect_dice"]) > float(best_result["defect_dice"])
        ):
            return True

    return False


def save_history_csv(path, history_rows):
    fieldnames = [
        "epoch",
        "train_loss",
        "defect_dice",
        "defect_iou",
        "defect_image_recall",
        "normal_fpr",
        "threshold",
        "lr_encoder",
        "lr_decoder",
    ]
    write_csv_rows(path, history_rows, fieldnames)


def main():
    args = parse_args()

    cfg = load_yaml(resolve_path(args.config))

    fold = int(args.fold if args.fold is not None else cfg.get("fold", 0))
    cfg = apply_fold_overrides(cfg, fold)

    set_seed(int(cfg.get("seed", 42)))

    device = build_device(cfg)

    defect_train_rows = read_csv_rows(resolve_path(cfg["defect_train_manifest"]))
    defect_val_rows = read_csv_rows(resolve_path(cfg["defect_val_manifest"]))
    normal_train_rows = read_csv_rows(resolve_path(cfg["normal_train_manifest"]))
    normal_val_rows = read_csv_rows(resolve_path(cfg["normal_val_manifest"]))

    val_loader = build_stage2_val_loader(defect_val_rows, normal_val_rows, cfg, device)

    model = build_model(pretrained=bool(cfg.get("pretrained", False)))
    model.to(device)

    stage1_checkpoint_path = resolve_path(cfg["stage1_checkpoint"])
    load_checkpoint(stage1_checkpoint_path, model, map_location=device)

    criterion = BCEDiceLoss(
        bce_weight=float(cfg.get("bce_weight", 0.5)),
        dice_weight=float(cfg.get("dice_weight", 0.5)),
        pos_weight=float(cfg.get("pos_weight", 12.0)),
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = build_scaler(cfg, device)

    early_stopper = EarlyStopper(
        patience=int(cfg.get("early_stop_patience", 12)),
        mode="max",
        min_delta=float(cfg.get("early_stop_min_delta", 0.0)),
    )

    epochs = int(cfg.get("epochs", 25))
    threshold = float(cfg.get("threshold", 0.5))

    save_dir = resolve_path(cfg["save_dir"])
    ensure_dir(save_dir)

    best_ckpt_path = save_dir / "best_stage2.pt"
    last_ckpt_path = save_dir / "last_stage2.pt"
    history_path = save_dir / "history.csv"

    history_rows = []
    best_stage2_result = None

    print(f"fold = {fold}")
    print(f"device = {device}")
    print(f"stage1_checkpoint = {stage1_checkpoint_path}")
    print(f"defect_train_count = {len(defect_train_rows)}")
    print(f"defect_val_count = {len(defect_val_rows)}")
    print(f"normal_train_count = {len(normal_train_rows)}")
    print(f"normal_val_count = {len(normal_val_rows)}")
    print(f"save_dir = {save_dir}")

    for epoch_index in range(epochs):
        epoch = epoch_index + 1
        epoch_seed = int(cfg.get("seed", 42)) + epoch

        epoch_train_rows = build_epoch_train_rows(defect_train_rows, normal_train_rows, epoch_seed)
        train_loader = build_stage2_train_loader(epoch_train_rows, cfg, device)

        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)

        val_stats = validate_stage2(
            model=model,
            loader=val_loader,
            device=device,
            threshold=threshold,
        )

        scheduler.step(float(val_stats["defect_dice"]))
        should_stop, _ = early_stopper.step(float(val_stats["defect_dice"]))

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "defect_dice": val_stats["defect_dice"],
                "defect_iou": val_stats["defect_iou"],
                "defect_image_recall": val_stats["defect_image_recall"],
                "normal_fpr": val_stats["normal_fpr"],
                "threshold": val_stats["threshold"],
                "lr_encoder": train_stats["lr_encoder"],
                "lr_decoder": train_stats["lr_decoder"],
            }
        )
        save_history_csv(history_path, history_rows)

        current_is_best = stage2_result_is_better(val_stats, best_stage2_result)

        if current_is_best:
            best_stage2_result = {
                "defect_dice": float(val_stats["defect_dice"]),
                "defect_iou": float(val_stats["defect_iou"]),
                "defect_image_recall": float(val_stats["defect_image_recall"]),
                "normal_fpr": float(val_stats["normal_fpr"]),
                "threshold": float(val_stats["threshold"]),
            }

        checkpoint = {
            "epoch": epoch,
            "fold": fold,
            "config": cfg,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "early_stopper_state_dict": early_stopper.state_dict(),
            "best_stage2_result": best_stage2_result,
            "current_val_stats": {
                "defect_dice": val_stats["defect_dice"],
                "defect_iou": val_stats["defect_iou"],
                "defect_image_recall": val_stats["defect_image_recall"],
                "normal_fpr": val_stats["normal_fpr"],
                "threshold": val_stats["threshold"],
            },
        }

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        save_checkpoint(last_ckpt_path, checkpoint)

        if current_is_best:
            save_checkpoint(best_ckpt_path, checkpoint)

        print(
            f"epoch {epoch}/{epochs} | "
            f"train_loss={train_stats['loss']:.6f} | "
            f"defect_dice={val_stats['defect_dice']:.6f} | "
            f"defect_iou={val_stats['defect_iou']:.6f} | "
            f"defect_recall={val_stats['defect_image_recall']:.6f} | "
            f"normal_fpr={val_stats['normal_fpr']:.6f} | "
            f"thr={val_stats['threshold']:.2f} | "
            f"lr_encoder={train_stats['lr_encoder']:.6e} | "
            f"lr_decoder={train_stats['lr_decoder']:.6e}"
        )

        if should_stop:
            print("early stopping 触发，stage2 提前结束。")
            break

    print("stage2 训练结束。")


if __name__ == "__main__":
    main()
