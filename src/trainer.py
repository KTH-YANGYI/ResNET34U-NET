from pathlib import Path

import torch

from src.utils import ensure_dir


def get_cfg_value(cfg, key, default_value):
    """
    从配置字典里取值。
    如果没有这个 key，就返回默认值。
    """

    if key in cfg:
        return cfg[key]

    return default_value


def get_param_group_lr(optimizer, group_name):
    """
    读取 optimizer 里某个参数组当前的学习率。
    """

    for group in optimizer.param_groups:
        if group.get("name", "") == group_name:
            return float(group["lr"])

    raise ValueError(f"没有找到名字为 {group_name} 的参数组")


def build_optimizer(model, cfg):
    """
    构建优化器。

    这里把参数分成两组：
    1. encoder
    2. decoder

    这样后面可以给它们不同学习率。
    """

    encoder_lr = float(get_cfg_value(cfg, "encoder_lr", 1e-4))
    decoder_lr = float(get_cfg_value(cfg, "decoder_lr", 1e-4))
    weight_decay = float(get_cfg_value(cfg, "weight_decay", 0.0))
    optimizer_name = str(get_cfg_value(cfg, "optimizer", "adamw")).lower()

    encoder_params = [p for p in model.encoder_parameters() if p.requires_grad]
    decoder_params = [p for p in model.decoder_parameters() if p.requires_grad]

    param_groups = []

    if len(encoder_params) > 0:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
                "name": "encoder",
            }
        )

    if len(decoder_params) > 0:
        param_groups.append(
            {
                "params": decoder_params,
                "lr": decoder_lr,
                "weight_decay": weight_decay,
                "name": "decoder",
            }
        )

    if len(param_groups) == 0:
        raise ValueError("没有可训练参数，无法构建 optimizer")

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(param_groups)
    else:
        raise ValueError(f"暂不支持的 optimizer: {optimizer_name}")

    return optimizer


def build_scheduler(optimizer, cfg):
    """
    构建学习率调度器。

    这里使用 ReduceLROnPlateau，
    它会在验证指标一段时间不提升时自动降低学习率。
    """

    factor = float(get_cfg_value(cfg, "lr_factor", 0.5))
    patience = int(get_cfg_value(cfg, "lr_patience", 3))
    min_lr = float(get_cfg_value(cfg, "min_lr", 1e-7))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )

    return scheduler


def save_checkpoint(path, state):
    """
    保存 checkpoint。
    """

    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    """
    从 checkpoint 恢复模型参数，以及可选的 optimizer / scheduler 状态。
    """

    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def batch_to_device(batch, device):
    """
    把一个 batch 里的 image 和 mask 搬到指定设备。
    """

    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)

    return images, masks


def compute_binary_dice_from_logits(logits, target, threshold=0.5, eps=1e-6):
    """
    计算“验证指标意义上的 Dice”。

    注意这不是 loss，而是一个汇报效果的指标：
    1. 先对 logits 做 sigmoid
    2. 再做二值化
    3. 再和 GT 比较
    """

    if target.ndim == 3:
        target = target.unsqueeze(1)

    target = target.float()

    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()

    dims = (1, 2, 3)

    intersection = (pred * target).sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    target_sum = target.sum(dim=dims)

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)

    both_empty = (pred_sum == 0) & (target_sum == 0)
    dice = torch.where(both_empty, torch.ones_like(dice), dice)

    return dice.mean()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    训练一个完整 epoch。

    返回：
        {
            "loss": ...,
            "lr_encoder": ...,
            "lr_decoder": ...,
        }
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    use_amp = (
        device.type == "cuda"
        and scaler is not None
        and scaler.is_enabled()
    )

    for batch in loader:
        images, masks = batch_to_device(batch, device)
        batch_size = images.shape[0]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if not torch.isfinite(loss):
            raise ValueError(f"训练时出现非有限 loss: {float(loss.detach())}")

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("训练 loader 是空的，无法完成一个 epoch")

    avg_loss = total_loss / total_samples

    return {
        "loss": avg_loss,
        "lr_encoder": get_param_group_lr(optimizer, "encoder"),
        "lr_decoder": get_param_group_lr(optimizer, "decoder"),
    }


def extract_batch_value(batch, key, index):
    """
    从 DataLoader 拼出来的 batch 中，提取第 index 个样本的某个字段。
    """

    if key not in batch:
        return None

    value = batch[key]

    if torch.is_tensor(value):
        return value[index]

    if isinstance(value, (list, tuple)):
        return value[index]

    return value


def predict_on_loader(model, loader, device):
    """
    在一个 loader 上统一做推理。

    返回：
        一个列表，每个元素都是一个样本的预测结果字典。
    """

    model.eval()
    results = []

    with torch.no_grad():
        for batch in loader:
            images, masks = batch_to_device(batch, device)
            logits = model(images)

            batch_size = logits.shape[0]

            for index in range(batch_size):
                results.append(
                    {
                        "logits": logits[index].detach().cpu(),
                        "mask": masks[index].detach().cpu(),
                        "sample_id": extract_batch_value(batch, "sample_id", index),
                        "image_name": extract_batch_value(batch, "image_name", index),
                        "video_id": extract_batch_value(batch, "video_id", index),
                        "sample_type": extract_batch_value(batch, "sample_type", index),
                        "is_labeled": extract_batch_value(batch, "is_labeled", index),
                        "patch_id": extract_batch_value(batch, "patch_id", index),
                        "base_sample_id": extract_batch_value(batch, "base_sample_id", index),
                        "patch_type": extract_batch_value(batch, "patch_type", index),
                    }
                )

    return results


def validate_stage1(model, loader, criterion, device):
    """
    在 stage1 patch 验证集上做验证。

    返回：
        {
            "val_loss": ...,
            "patch_dice": ...,
        }
    """

    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images, masks = batch_to_device(batch, device)
            batch_size = images.shape[0]

            logits = model(images)
            loss = criterion(logits, masks)
            dice = compute_binary_dice_from_logits(logits, masks)

            total_loss += float(loss.detach()) * batch_size
            total_dice += float(dice.detach()) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("验证 loader 是空的，无法计算 stage1 指标")

    return {
        "val_loss": total_loss / total_samples,
        "patch_dice": total_dice / total_samples,
    }


def validate_stage2(model, loader, device, threshold=0.5):
    """
    在 stage2 整图验证集上做验证。

    这里直接用固定阈值二值化，不做额外后处理。
    """

    from src.metrics import evaluate_prob_maps, logits_to_probs

    predictions = predict_on_loader(model, loader, device)

    if len(predictions) == 0:
        raise ValueError("验证 loader 是空的，无法计算 stage2 指标")

    prob_maps = []
    gt_masks = []
    sample_types = []
    image_names = []

    for item in predictions:
        prob_map = logits_to_probs(item["logits"]).squeeze()
        gt_mask = item["mask"].squeeze().numpy()

        prob_maps.append(prob_map)
        gt_masks.append(gt_mask)
        sample_types.append(item.get("sample_type", ""))
        image_names.append(item.get("image_name", ""))

    result = evaluate_prob_maps(
        prob_maps=prob_maps,
        gt_masks=gt_masks,
        sample_types=sample_types,
        image_names=image_names,
        threshold=threshold,
    )

    return result


class EarlyStopper:
    """
    简单 early stopping。

    默认使用 "max" 模式，适合 patch_dice / defect_dice 这类指标。
    """

    def __init__(self, patience, mode="max", min_delta=0.0):
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)

        self.best_score = None
        self.num_bad_epochs = 0

    def is_improvement(self, score):
        if self.best_score is None:
            return True

        if self.mode == "max":
            return score > self.best_score + self.min_delta

        if self.mode == "min":
            return score < self.best_score - self.min_delta

        raise ValueError(f"不支持的 mode: {self.mode}")

    def step(self, score):
        """
        输入当前指标，返回：
            should_stop, is_best
        """

        if self.is_improvement(score):
            self.best_score = float(score)
            self.num_bad_epochs = 0
            return False, True

        self.num_bad_epochs += 1
        should_stop = self.num_bad_epochs >= self.patience

        return should_stop, False

    def state_dict(self):
        return {
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best_score": self.best_score,
            "num_bad_epochs": self.num_bad_epochs,
        }

    def load_state_dict(self, state):
        self.patience = state["patience"]
        self.mode = state["mode"]
        self.min_delta = state["min_delta"]
        self.best_score = state["best_score"]
        self.num_bad_epochs = state["num_bad_epochs"]
