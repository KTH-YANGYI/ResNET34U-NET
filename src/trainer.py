import sys
from contextlib import nullcontext
from pathlib import Path

import torch

from src.losses import get_primary_logits
from src.utils import ensure_dir

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


class _NullProgress:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        return None

    def close(self):
        return None


def build_progress(iterable, desc=None, leave=False):
    if tqdm is None or not sys.stderr.isatty():
        return _NullProgress(iterable)

    return tqdm(iterable, desc=desc, leave=leave, dynamic_ncols=True)


def get_cfg_value(cfg, key, default_value):
    if key in cfg:
        return cfg[key]

    return default_value


def get_param_group_lr(optimizer, group_name):
    for group in optimizer.param_groups:
        if group.get("name", "") == group_name:
            return float(group["lr"])

    raise ValueError(f"Could not find optimizer param group named {group_name}")


def build_optimizer(model, cfg):
    encoder_lr = float(get_cfg_value(cfg, "encoder_lr", 1e-4))
    decoder_lr = float(get_cfg_value(cfg, "decoder_lr", 1e-4))
    weight_decay = float(get_cfg_value(cfg, "weight_decay", 0.0))
    optimizer_name = str(get_cfg_value(cfg, "optimizer", "adamw")).lower()

    encoder_params = [param for param in model.encoder_parameters() if param.requires_grad]
    decoder_params = [param for param in model.decoder_parameters() if param.requires_grad]

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
        raise ValueError("No trainable parameters found for optimizer")

    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups)

    if optimizer_name == "adam":
        return torch.optim.Adam(param_groups)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer, cfg):
    factor = float(get_cfg_value(cfg, "lr_factor", 0.5))
    patience = int(get_cfg_value(cfg, "lr_patience", 3))
    min_lr = float(get_cfg_value(cfg, "min_lr", 1e-7))

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )


def save_checkpoint(path, state):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu", strict=True):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"], strict=bool(strict))

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def load_compatible_checkpoint(path, model, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    source_state = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []

    for key, value in source_state.items():
        if key not in model_state:
            skipped.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append(key)
            continue
        compatible_state[key] = value

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    print(
        "compatible checkpoint load | "
        f"loaded={len(compatible_state)} | "
        f"skipped={len(skipped)} | "
        f"missing={len(missing)} | "
        f"unexpected={len(unexpected)}"
    )
    if len(skipped) > 0:
        print("first skipped keys:", ", ".join(skipped[:20]))
    if len(missing) > 0:
        print("first missing keys:", ", ".join(list(missing)[:20]))
    if len(unexpected) > 0:
        print("first unexpected keys:", ", ".join(list(unexpected)[:20]))

    return checkpoint


def build_amp_grad_scaler(device, enabled=True):
    enabled = bool(enabled) and device.type == "cuda"
    if not enabled:
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device="cuda", enabled=True)
        except TypeError:
            try:
                return torch.amp.GradScaler("cuda", enabled=True)
            except TypeError:
                return torch.amp.GradScaler(enabled=True)

    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=True)

    return None


def amp_autocast(device, enabled=True):
    enabled = bool(enabled) and device.type == "cuda"
    if not enabled:
        return nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type=device.type, enabled=True)
        except TypeError:
            try:
                return torch.amp.autocast(device.type, enabled=True)
            except TypeError:
                pass

    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=True)

    return nullcontext()


def batch_to_device(batch, device):
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    return images, masks


def _prepare_target(target):
    if target.ndim == 3:
        target = target.unsqueeze(1)
    return target.float()


def compute_binary_dice_per_sample_from_logits(logits, target, threshold=0.5, eps=1e-6):
    target = _prepare_target(target)
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()

    dims = (1, 2, 3)
    intersection = (pred * target).sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    target_sum = target.sum(dim=dims)

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    both_empty = (pred_sum == 0) & (target_sum == 0)
    dice = torch.where(both_empty, torch.ones_like(dice), dice)

    return {
        "dice": dice,
        "pred_sum": pred_sum,
        "target_sum": target_sum,
    }


def compute_binary_dice_from_logits(logits, target, threshold=0.5, eps=1e-6):
    return compute_binary_dice_per_sample_from_logits(logits, target, threshold=threshold, eps=eps)["dice"].mean()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, progress_desc=None):
    model.train()
    if hasattr(model, "apply_encoder_freeze_mode"):
        model.apply_encoder_freeze_mode()

    total_loss = 0.0
    total_samples = 0

    use_amp = device.type == "cuda" and scaler is not None and scaler.is_enabled()

    progress = build_progress(loader, desc=progress_desc, leave=False)

    try:
        for batch in progress:
            images, masks = batch_to_device(batch, device)
            batch_size = images.shape[0]

            optimizer.zero_grad(set_to_none=True)

            with amp_autocast(device, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            if not torch.isfinite(loss):
                raise ValueError(f"Training produced non-finite loss: {float(loss.detach())}")

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size
            progress.set_postfix(
                batch_loss=f"{float(loss.detach()):.4f}",
                avg_loss=f"{(total_loss / total_samples):.4f}",
            )
    finally:
        progress.close()

    if total_samples == 0:
        raise ValueError("Training loader is empty")

    return {
        "loss": total_loss / total_samples,
        "lr_encoder": get_param_group_lr(optimizer, "encoder"),
        "lr_decoder": get_param_group_lr(optimizer, "decoder"),
    }


def extract_batch_value(batch, key, index):
    if key not in batch:
        return None

    value = batch[key]

    if torch.is_tensor(value):
        return value[index]

    if isinstance(value, (list, tuple)):
        return value[index]

    return value


def predict_on_loader(model, loader, device, progress_desc=None, amp_enabled=True):
    model.eval()
    results = []
    use_amp = bool(amp_enabled) and device.type == "cuda"

    progress = build_progress(loader, desc=progress_desc, leave=False)

    with torch.no_grad():
        try:
            for batch in progress:
                images, masks = batch_to_device(batch, device)
                with amp_autocast(device, enabled=use_amp):
                    logits = get_primary_logits(model(images))
                batch_size = logits.shape[0]

                for index in range(batch_size):
                    results.append(
                        {
                            "logits": logits[index].detach().cpu(),
                            "mask": masks[index].detach().cpu(),
                            "sample_id": extract_batch_value(batch, "sample_id", index),
                            "image_name": extract_batch_value(batch, "image_name", index),
                            "video_id": extract_batch_value(batch, "video_id", index),
                            "video_name": extract_batch_value(batch, "video_name", index),
                            "frame_id": extract_batch_value(batch, "frame_id", index),
                            "sample_type": extract_batch_value(batch, "sample_type", index),
                            "source_split": extract_batch_value(batch, "source_split", index),
                            "device": extract_batch_value(batch, "device", index),
                            "defect_class": extract_batch_value(batch, "defect_class", index),
                            "holdout_reason": extract_batch_value(batch, "holdout_reason", index),
                            "is_labeled": extract_batch_value(batch, "is_labeled", index),
                            "patch_id": extract_batch_value(batch, "patch_id", index),
                            "base_sample_id": extract_batch_value(batch, "base_sample_id", index),
                            "patch_type": extract_batch_value(batch, "patch_type", index),
                            "patch_family": extract_batch_value(batch, "patch_family", index),
                        }
                    )
        finally:
            progress.close()

    return results


def validate_stage1(model, loader, criterion, device, threshold=0.5, progress_desc=None, amp_enabled=True):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    per_patch_rows = []
    use_amp = bool(amp_enabled) and device.type == "cuda"

    progress = build_progress(loader, desc=progress_desc, leave=False)

    with torch.no_grad():
        try:
            for batch in progress:
                images, masks = batch_to_device(batch, device)
                batch_size = images.shape[0]

                with amp_autocast(device, enabled=use_amp):
                    output = model(images)
                    loss = criterion(output, masks)
                    logits = get_primary_logits(output)
                dice_info = compute_binary_dice_per_sample_from_logits(logits, masks, threshold=threshold)

                total_loss += float(loss.detach()) * batch_size
                total_samples += batch_size
                progress.set_postfix(avg_loss=f"{(total_loss / total_samples):.4f}")

                for index in range(batch_size):
                    dice_value = float(dice_info["dice"][index].detach().cpu())
                    pred_sum = float(dice_info["pred_sum"][index].detach().cpu())
                    target_sum = float(dice_info["target_sum"][index].detach().cpu())
                    pred_has_positive = pred_sum > 0.0
                    target_has_positive = target_sum > 0.0

                    per_patch_rows.append(
                        {
                            "patch_type": str(extract_batch_value(batch, "patch_type", index) or ""),
                            "patch_family": str(extract_batch_value(batch, "patch_family", index) or ""),
                            "dice": dice_value,
                            "pred_has_positive": pred_has_positive,
                            "target_has_positive": target_has_positive,
                        }
                    )
        finally:
            progress.close()

    if total_samples == 0:
        raise ValueError("Validation loader is empty")

    patch_dice_all = float(sum(item["dice"] for item in per_patch_rows) / len(per_patch_rows))

    positive_rows = [item for item in per_patch_rows if item["target_has_positive"] or item["patch_family"] == "positive"]
    negative_rows = [item for item in per_patch_rows if not item["target_has_positive"] and item["patch_family"] != "positive"]

    if len(positive_rows) > 0:
        patch_dice_pos_only = float(sum(item["dice"] for item in positive_rows) / len(positive_rows))
        positive_patch_recall = float(sum(1.0 for item in positive_rows if item["pred_has_positive"]) / len(positive_rows))
    else:
        patch_dice_pos_only = 0.0
        positive_patch_recall = 0.0

    if len(negative_rows) > 0:
        negative_patch_fpr = float(sum(1.0 for item in negative_rows if item["pred_has_positive"]) / len(negative_rows))
    else:
        negative_patch_fpr = 0.0

    patch_type_to_dice = {}
    count_by_type = {}

    for item in per_patch_rows:
        patch_type = item["patch_type"]
        patch_type_to_dice.setdefault(patch_type, []).append(item["dice"])
        count_by_type[patch_type] = count_by_type.get(patch_type, 0) + 1

    patch_dice_by_type = {
        patch_type: float(sum(values) / len(values))
        for patch_type, values in sorted(patch_type_to_dice.items(), key=lambda item: item[0])
    }

    return {
        "val_loss": total_loss / total_samples,
        "patch_dice": patch_dice_all,
        "patch_dice_all": patch_dice_all,
        "patch_dice_pos_only": patch_dice_pos_only,
        "positive_patch_recall": positive_patch_recall,
        "negative_patch_fpr": negative_patch_fpr,
        "patch_dice_by_type": patch_dice_by_type,
        "count_by_type": dict(sorted(count_by_type.items(), key=lambda item: item[0])),
    }


def validate_stage2(
    model,
    loader,
    device,
    threshold=0.5,
    min_area=0,
    threshold_values=None,
    min_area_values=None,
    target_normal_fpr=0.10,
    lambda_fpr_penalty=2.0,
    progress_desc=None,
    amp_enabled=True,
    include_auprc=False,
):
    from src.metrics import compute_stage2_score, evaluate_prob_maps, logits_to_probs, search_postprocess_params

    predictions = predict_on_loader(model, loader, device, progress_desc=progress_desc, amp_enabled=amp_enabled)

    if len(predictions) == 0:
        raise ValueError("Validation loader is empty")

    prob_maps = []
    gt_masks = []
    sample_types = []
    image_names = []

    for item in predictions:
        prob_maps.append(logits_to_probs(item["logits"]).squeeze())
        gt_masks.append(item["mask"].squeeze().numpy())
        sample_types.append(item.get("sample_type", ""))
        image_names.append(item.get("image_name", ""))

    if threshold_values is None:
        threshold_values = [float(threshold)]

    if min_area_values is None:
        min_area_values = [int(min_area)]

    threshold_values = [float(value) for value in threshold_values]
    min_area_values = [int(value) for value in min_area_values]

    if len(threshold_values) == 1 and len(min_area_values) == 1:
        result = evaluate_prob_maps(
            prob_maps=prob_maps,
            gt_masks=gt_masks,
            sample_types=sample_types,
            image_names=image_names,
            threshold=threshold_values[0],
            min_area=min_area_values[0],
            include_auprc=include_auprc,
        )
        result["stage2_score"] = compute_stage2_score(
            result,
            target_normal_fpr=target_normal_fpr,
            lambda_fpr_penalty=lambda_fpr_penalty,
        )
        result["search_rows"] = []
        return result

    search_output = search_postprocess_params(
        prob_maps=prob_maps,
        gt_masks=gt_masks,
        sample_types=sample_types,
        image_names=image_names,
        threshold_values=threshold_values,
        min_area_values=min_area_values,
        target_normal_fpr=target_normal_fpr,
        lambda_fpr_penalty=lambda_fpr_penalty,
        include_auprc=include_auprc,
    )

    best_result = search_output["best_result"]
    best_result["search_rows"] = search_output["search_rows"]
    return best_result


class EarlyStopper:
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
            return float(score) > float(self.best_score) + self.min_delta

        if self.mode == "min":
            return float(score) < float(self.best_score) - self.min_delta

        raise ValueError(f"Unsupported mode: {self.mode}")

    def step(self, score):
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
