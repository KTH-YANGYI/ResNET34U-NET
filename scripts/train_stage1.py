import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets import (
    PatchDataset,
    build_stage1_eval_transform,
    build_stage1_train_transform,
)
from src.losses import BCEDiceLoss
from src.model import build_model
from src.trainer import (
    EarlyStopper,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    train_one_epoch,
    validate_stage1,
)
from src.utils import ensure_dir, load_yaml, read_csv_rows, seed_worker, set_seed, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Train stage1 patch model")
    parser.add_argument("--config", type=str, default="configs/stage1.yaml", help="配置文件路径")
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

    if "train_index_path_template" in cfg:
        cfg["train_index_path"] = str(cfg["train_index_path_template"]).format(fold=fold)

    if "val_index_path_template" in cfg:
        cfg["val_index_path"] = str(cfg["val_index_path_template"]).format(fold=fold)

    if "save_dir_template" in cfg:
        cfg["save_dir"] = str(cfg["save_dir_template"]).format(fold=fold)

    return cfg


def build_stage1_loaders(cfg, device):
    train_rows = read_csv_rows(resolve_path(cfg["train_index_path"]))
    val_rows = read_csv_rows(resolve_path(cfg["val_index_path"]))

    patch_out_size = int(cfg.get("patch_out_size", 384))
    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 8))
    seed = int(cfg.get("seed", 42))

    train_dataset = PatchDataset(
        train_rows,
        transform=build_stage1_train_transform(patch_out_size),
    )
    val_dataset = PatchDataset(
        val_rows,
        transform=build_stage1_eval_transform(patch_out_size),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = device.type == "cuda"
    worker_init_fn = seed_worker if num_workers > 0 else None
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, len(train_rows), len(val_rows)


def build_scaler(cfg, device):
    amp_enabled = bool(cfg.get("amp", True))
    use_amp = amp_enabled and device.type == "cuda"

    if not use_amp:
        return None

    return torch.amp.GradScaler(device="cuda", enabled=True)


def save_history_csv(path, history_rows):
    fieldnames = [
        "epoch",
        "encoder_frozen",
        "train_loss",
        "val_loss",
        "patch_dice",
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
    train_loader, val_loader, train_patch_count, val_patch_count = build_stage1_loaders(cfg, device)

    model = build_model(pretrained=bool(cfg.get("pretrained", True)))
    model.to(device)

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

    epochs = int(cfg.get("epochs", 30))
    freeze_encoder_epochs = int(cfg.get("freeze_encoder_epochs", 3))

    save_dir = resolve_path(cfg["save_dir"])
    ensure_dir(save_dir)

    best_ckpt_path = save_dir / "best_stage1.pt"
    last_ckpt_path = save_dir / "last_stage1.pt"
    history_path = save_dir / "history.csv"

    history_rows = []

    print(f"fold = {fold}")
    print(f"device = {device}")
    print(f"train_patch_count = {train_patch_count}")
    print(f"val_patch_count = {val_patch_count}")
    print(f"save_dir = {save_dir}")

    for epoch_index in range(epochs):
        epoch = epoch_index + 1

        encoder_trainable = epoch_index >= freeze_encoder_epochs
        model.set_encoder_trainable(encoder_trainable)

        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_stats = validate_stage1(model, val_loader, criterion, device)

        current_patch_dice = float(val_stats["patch_dice"])
        scheduler.step(current_patch_dice)

        should_stop, is_best = early_stopper.step(current_patch_dice)

        history_rows.append(
            {
                "epoch": epoch,
                "encoder_frozen": not encoder_trainable,
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["val_loss"],
                "patch_dice": val_stats["patch_dice"],
                "lr_encoder": train_stats["lr_encoder"],
                "lr_decoder": train_stats["lr_decoder"],
            }
        )
        save_history_csv(history_path, history_rows)

        checkpoint = {
            "epoch": epoch,
            "fold": fold,
            "config": cfg,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "early_stopper_state_dict": early_stopper.state_dict(),
            "best_patch_dice": early_stopper.best_score,
        }

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        save_checkpoint(last_ckpt_path, checkpoint)

        if is_best:
            save_checkpoint(best_ckpt_path, checkpoint)

        print(
            f"epoch {epoch}/{epochs} | "
            f"encoder_frozen={not encoder_trainable} | "
            f"train_loss={train_stats['loss']:.6f} | "
            f"val_loss={val_stats['val_loss']:.6f} | "
            f"patch_dice={val_stats['patch_dice']:.6f} | "
            f"lr_encoder={train_stats['lr_encoder']:.6e} | "
            f"lr_decoder={train_stats['lr_decoder']:.6e}"
        )

        if should_stop:
            print("early stopping 触发，stage1 提前结束。")
            break

    print("stage1 训练结束。")


if __name__ == "__main__":
    main()
