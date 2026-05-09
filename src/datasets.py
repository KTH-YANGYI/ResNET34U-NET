import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def to_bool(value):
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def normalize_size(size):
    if isinstance(size, int):
        return size, size

    if isinstance(size, (tuple, list)) and len(size) == 2:
        return int(size[0]), int(size[1])

    raise ValueError("size must be an int or a (height, width) pair")


def read_image_rgb(path):
    image = Image.open(Path(path)).convert("RGB")
    return np.array(image)


def read_mask_binary(path, image_size=None):
    if str(path).strip() == "":
        raise ValueError("read_mask_binary received an empty path")

    mask = Image.open(Path(path)).convert("L")

    if image_size is not None:
        target_h, target_w = normalize_size(image_size)
        mask = mask.resize((target_w, target_h), resample=Image.NEAREST)

    mask = np.array(mask)
    return (mask > 0).astype(np.uint8)


def build_empty_mask(height, width):
    return np.zeros((height, width), dtype=np.uint8)


def resize_image(image, target_size):
    target_h, target_w = normalize_size(target_size)
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.array(pil_image)


def resize_mask(mask, target_size):
    target_h, target_w = normalize_size(target_size)
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    pil_mask = Image.fromarray(mask_uint8)
    pil_mask = pil_mask.resize((target_w, target_h), resample=Image.NEAREST)
    return (np.array(pil_mask) > 0).astype(np.uint8)


def image_to_tensor(image):
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(image)


def mask_to_tensor(mask):
    mask = mask.astype(np.float32)[None, :, :]
    return torch.from_numpy(mask)


def normalize_image_tensor(image_tensor, use_imagenet_normalize=False):
    if not use_imagenet_normalize:
        return image_tensor

    return (image_tensor - IMAGENET_MEAN) / IMAGENET_STD


class BasicSegTransform:
    def __init__(self, target_size, use_imagenet_normalize=False):
        self.target_size = target_size
        self.use_imagenet_normalize = bool(use_imagenet_normalize)

    def finalize(self, image, mask):
        image = resize_image(image, self.target_size)
        mask = resize_mask(mask, self.target_size)

        image_tensor = image_to_tensor(image)
        image_tensor = normalize_image_tensor(image_tensor, use_imagenet_normalize=self.use_imagenet_normalize)
        mask_tensor = mask_to_tensor(mask)
        return image_tensor, mask_tensor

    def __call__(self, image, mask):
        return self.finalize(image, mask)


class TrainAugSegTransform(BasicSegTransform):
    def __init__(
        self,
        target_size,
        use_imagenet_normalize=False,
        hflip_p=0.0,
        vflip_p=0.0,
        rotate_deg=0.0,
        brightness=0.0,
        contrast=0.0,
        gamma=0.0,
        noise_std=0.0,
        blur_p=0.0,
    ):
        super().__init__(target_size=target_size, use_imagenet_normalize=use_imagenet_normalize)
        self.hflip_p = float(hflip_p)
        self.vflip_p = float(vflip_p)
        self.rotate_deg = float(rotate_deg)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.gamma = float(gamma)
        self.noise_std = float(noise_std)
        self.blur_p = float(blur_p)

    def __call__(self, image, mask):
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)

        if self.hflip_p > 0.0 and random.random() < self.hflip_p:
            image_pil = ImageOps.mirror(image_pil)
            mask_pil = ImageOps.mirror(mask_pil)

        if self.vflip_p > 0.0 and random.random() < self.vflip_p:
            image_pil = ImageOps.flip(image_pil)
            mask_pil = ImageOps.flip(mask_pil)

        if self.rotate_deg > 0.0:
            angle = random.uniform(-self.rotate_deg, self.rotate_deg)
            image_pil = image_pil.rotate(angle, resample=Image.BILINEAR)
            mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST)

        if self.brightness > 0.0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image_pil = ImageEnhance.Brightness(image_pil).enhance(factor)

        if self.contrast > 0.0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image_pil = ImageEnhance.Contrast(image_pil).enhance(factor)

        image_np = np.array(image_pil)

        if self.gamma > 0.0:
            gamma_value = max(0.05, 1.0 + random.uniform(-self.gamma, self.gamma))
            image_np = np.clip((image_np.astype(np.float32) / 255.0) ** gamma_value, 0.0, 1.0)
            image_np = (image_np * 255.0).astype(np.uint8)

        if self.blur_p > 0.0 and random.random() < self.blur_p:
            radius = random.uniform(0.2, 1.2)
            image_np = np.array(Image.fromarray(image_np).filter(ImageFilter.GaussianBlur(radius=radius)))

        if self.noise_std > 0.0:
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=image_np.shape).astype(np.float32)
            image_np = np.clip(image_np.astype(np.float32) / 255.0 + noise, 0.0, 1.0)
            image_np = (image_np * 255.0).astype(np.uint8)

        mask_np = (np.array(mask_pil) > 0).astype(np.uint8)
        return self.finalize(image_np, mask_np)


def _cfg_bool(cfg, key, default_value):
    if cfg is None:
        return bool(default_value)

    return bool(cfg.get(key, default_value))


def build_stage1_train_transform(out_size, cfg=None):
    use_imagenet_normalize = _cfg_bool(cfg, "use_imagenet_normalize", False)
    augment_enable = _cfg_bool(cfg, "augment_enable", False)

    if not augment_enable:
        return BasicSegTransform(out_size, use_imagenet_normalize=use_imagenet_normalize)

    return TrainAugSegTransform(
        target_size=out_size,
        use_imagenet_normalize=use_imagenet_normalize,
        hflip_p=float(cfg.get("augment_hflip_p", 0.5)),
        vflip_p=float(cfg.get("augment_vflip_p", 0.5)),
        rotate_deg=float(cfg.get("augment_rotate_deg", 15.0)),
        brightness=float(cfg.get("augment_brightness", 0.15)),
        contrast=float(cfg.get("augment_contrast", 0.15)),
        gamma=float(cfg.get("augment_gamma", 0.15)),
        noise_std=float(cfg.get("augment_noise_std", 0.02)),
        blur_p=float(cfg.get("augment_blur_p", 0.15)),
    )


def build_stage1_eval_transform(out_size, cfg=None):
    use_imagenet_normalize = _cfg_bool(cfg, "use_imagenet_normalize", False)
    return BasicSegTransform(out_size, use_imagenet_normalize=use_imagenet_normalize)


def build_stage2_train_transform(image_size, cfg=None):
    use_imagenet_normalize = _cfg_bool(cfg, "use_imagenet_normalize", False)
    augment_enable = _cfg_bool(cfg, "augment_enable", False)

    if not augment_enable:
        return BasicSegTransform(image_size, use_imagenet_normalize=use_imagenet_normalize)

    return TrainAugSegTransform(
        target_size=image_size,
        use_imagenet_normalize=use_imagenet_normalize,
        hflip_p=float(cfg.get("augment_hflip_p", 0.5)),
        vflip_p=float(cfg.get("augment_vflip_p", 0.5)),
        rotate_deg=float(cfg.get("augment_rotate_deg", 10.0)),
        brightness=float(cfg.get("augment_brightness", 0.10)),
        contrast=float(cfg.get("augment_contrast", 0.10)),
        gamma=float(cfg.get("augment_gamma", 0.0)),
        noise_std=float(cfg.get("augment_noise_std", 0.015)),
        blur_p=float(cfg.get("augment_blur_p", 0.0)),
    )


def build_stage2_eval_transform(image_size, cfg=None):
    use_imagenet_normalize = _cfg_bool(cfg, "use_imagenet_normalize", False)
    return BasicSegTransform(image_size, use_imagenet_normalize=use_imagenet_normalize)


def crop_patch(image, mask, x, y, crop_size, out_size):
    del out_size

    x = int(x)
    y = int(y)
    crop_size = int(crop_size)

    image_h, image_w = image.shape[:2]
    x = max(0, min(x, image_w - crop_size))
    y = max(0, min(y, image_h - crop_size))

    image_patch = image[y:y + crop_size, x:x + crop_size]
    mask_patch = mask[y:y + crop_size, x:x + crop_size]
    return image_patch, mask_patch


class ROIDataset(Dataset):
    def __init__(self, rows, image_size, transform=None, return_empty_mask_for_unlabeled=True):
        self.rows = list(rows)
        self.image_size = image_size
        self.return_empty_mask_for_unlabeled = bool(return_empty_mask_for_unlabeled)
        self.transform = transform if transform is not None else build_stage2_eval_transform(image_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = read_image_rgb(row["image_path"])
        original_h, original_w = image.shape[:2]

        mask_path = str(row.get("mask_path", "")).strip()
        if mask_path != "":
            mask = read_mask_binary(mask_path)
        elif self.return_empty_mask_for_unlabeled:
            mask = build_empty_mask(original_h, original_w)
        else:
            raise ValueError(f"Sample {row['image_name']} has no mask_path")

        image_tensor, mask_tensor = self.transform(image, mask)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": row["sample_id"],
            "image_name": row["image_name"],
            "video_id": str(row.get("video_id", "")).strip(),
            "video_name": str(row.get("video_name", "")).strip(),
            "frame_id": str(row.get("frame_id", "")).strip(),
            "sample_type": row.get("sample_type", ""),
            "source_split": row.get("source_split", ""),
            "device": row.get("device", ""),
            "defect_class": row.get("defect_class", ""),
            "holdout_reason": row.get("holdout_reason", ""),
            "is_labeled": to_bool(row.get("is_labeled", False)),
        }


class PatchDataset(Dataset):
    def __init__(self, patch_rows, transform=None, cache_enabled=False, cache_max_items=0):
        self.patch_rows = list(patch_rows)
        self.transform = transform
        self.cache_enabled = bool(cache_enabled)
        self.cache_max_items = int(cache_max_items)
        self._image_cache = {}
        self._mask_cache = {}

    def __len__(self):
        return len(self.patch_rows)

    def _read_cached(self, cache, key, read_fn):
        if not self.cache_enabled:
            return read_fn(key)

        key = str(key)
        if key in cache:
            return cache[key]

        value = read_fn(key)
        if self.cache_max_items <= 0 or len(cache) < self.cache_max_items:
            cache[key] = value
        return value

    def __getitem__(self, index):
        row = self.patch_rows[index]
        image = self._read_cached(self._image_cache, row["image_path"], read_image_rgb)

        mask_path = str(row.get("mask_path", "")).strip()
        if mask_path != "":
            mask = self._read_cached(self._mask_cache, mask_path, read_mask_binary)
        else:
            image_h, image_w = image.shape[:2]
            mask = build_empty_mask(image_h, image_w)

        crop_x = int(row["crop_x"])
        crop_y = int(row["crop_y"])
        crop_size = int(row["crop_size"])
        out_size = int(row["out_size"])

        image_patch, mask_patch = crop_patch(
            image=image,
            mask=mask,
            x=crop_x,
            y=crop_y,
            crop_size=crop_size,
            out_size=out_size,
        )

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_patch, mask_patch)
        else:
            patch_transform = build_stage1_eval_transform(out_size)
            image_tensor, mask_tensor = patch_transform(image_patch, mask_patch)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "patch_id": row.get("patch_id", ""),
            "base_sample_id": row.get("base_sample_id", ""),
            "patch_type": row.get("patch_type", ""),
            "patch_family": row.get("patch_family", ""),
            "video_id": str(row.get("video_id", "")).strip(),
            "video_name": str(row.get("video_name", "")).strip(),
            "frame_id": str(row.get("frame_id", "")).strip(),
            "sample_type": row.get("sample_type", ""),
            "source_split": row.get("source_split", ""),
            "component_id": row.get("component_id", ""),
            "component_area_px": row.get("component_area_px", ""),
            "is_replay": to_bool(row.get("is_replay", False)),
        }
