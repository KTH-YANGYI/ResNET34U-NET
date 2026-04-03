from pathlib import Path
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset


def to_bool(value):
    """
    把各种形式的 True / False 统一转成真正的布尔值。
     csv 读出来以后，is_labeled 往往会变成字符串：
        "True"
        "False"
    将它变成布尔值
    """

    if isinstance(value, bool):
        return value

    # 先把输入统一转成小写字符串，再去比较
    text = str(value).strip().lower()

    # 只要属于下面这些写法，就当作 True
    if text in {"1", "true", "yes", "y"}:
        return True

    return False


def normalize_size(size):
    """
    把输入的尺寸统一整理成 (height, width) 这种形式。
    """
    if isinstance(size, int):
        return size, size

    if isinstance(size, (tuple, list)) and len(size) == 2:
        return int(size[0]), int(size[1])

    # 如果既不是 int，也不是长度为 2 的 tuple/list，就报错
    raise ValueError("size 必须是 int，或者 (height, width)")


def read_image_rgb(path):
    """
    读取一张彩色图，并返回 numpy 数组。
    """

    path = Path(path)

    # convert("RGB") 表示强制转成 RGB 三通道
    image = Image.open(path).convert("RGB")

    image = np.array(image)

    return image


def read_mask_binary(path, image_size=None):
    """
    读取二值 mask，并把它变成 0/1 的 numpy 数组。
    """

    if str(path).strip() == "":
        raise ValueError("read_mask_binary 收到了空路径，说明这条样本没有真实 mask")

    path = Path(path)

    # convert("L") 表示读成单通道灰度图
    mask = Image.open(path).convert("L")

    # 如果指定了 image_size，就先缩放
    if image_size is not None:
        target_h, target_w = normalize_size(image_size)

        # PIL 的 resize 接收顺序是 (width, height)
        mask = mask.resize((target_w, target_h), resample=Image.NEAREST)

    # 转成 numpy 数组
    mask = np.array(mask)

    # 二值化：
    # 只要像素值 > 0，就当成前景 1 .astype(np.uint8) 把true转成1
    mask = (mask > 0).astype(np.uint8)

    return mask


def build_empty_mask(height, width):
    """
    创建一张全 0 的 mask。

    1. normal 样本没有真实缺陷 mask
    2. holdout 样本当前也没有标注 mask

    这时我们就返回一张全黑的 mask。
    """

    return np.zeros((height, width), dtype=np.uint8)


def resize_image(image, target_size):

    target_h, target_w = normalize_size(target_size)

    # 先把 numpy 数组转回 PIL 图片
    pil_image = Image.fromarray(image)

    # 彩色图缩放时，用 BILINEAR 会更平滑
    pil_image = pil_image.resize((target_w, target_h), resample=Image.BILINEAR)

    # 再转回 numpy
    image = np.array(pil_image)

    return image


def resize_mask(mask, target_size):
    """
    这个函数的作用：
    把 mask resize 到指定大小。

    注意：
    mask 和彩色图不一样。
    mask 不能用 BILINEAR，
    否则 0 和 1 会被插值成奇怪的小数。

    所以 mask 必须用 NEAREST 最近邻插值。
    """

    target_h, target_w = normalize_size(target_size)

    # 先把 0/1 mask 变成 0/255，方便转成图片
    mask_uint8 = (mask * 255).astype(np.uint8)

    pil_mask = Image.fromarray(mask_uint8)

    pil_mask = pil_mask.resize((target_w, target_h), resample=Image.NEAREST)

    mask = np.array(pil_mask)

    # resize 后再做一次二值化，保证最后还是严格的 0/1
    mask = (mask > 0).astype(np.uint8)

    return mask


def image_to_tensor(image):
    """
    把 numpy 彩色图转成 PyTorch tensor。

    同时还会把像素值从 0~255 归一化到 0~1。
    """

    # 把通道维从最后一维，挪到第一维
    image = image.transpose(2, 0, 1)

    # astype(np.float32) 转成 float32
    image = image.astype(np.float32) / 255.0

    # torch.from_numpy 把 numpy 数组转成 tensor
    image_tensor = torch.from_numpy(image)

    return image_tensor


def mask_to_tensor(mask):
    mask = mask.astype(np.float32)

    # H x W -> 1 x H x W
    mask = mask[None, :, :]

    mask_tensor = torch.from_numpy(mask)

    return mask_tensor


class BasicSegTransform:
    """
    这个类的作用：
    做最基础的分割预处理。

    当前第一版我们只做：
    1. resize
    2. 转 tensor

    暂时不加随机翻转、颜色扰动这些增强，
    因为我们现在先求“数据流跑通”。

    为什么这里不用内部嵌套函数，而是写成一个类？
    因为 Windows 下 DataLoader 多进程时，
    顶层类通常比局部函数更稳。
    """

    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image, mask):
        """
        当你写 transform(image, mask) 时，
        实际上调用的就是这个 __call__ 函数。
        """

        image = resize_image(image, self.target_size)
        mask = resize_mask(mask, self.target_size)

        image_tensor = image_to_tensor(image)
        mask_tensor = mask_to_tensor(mask)

        return image_tensor, mask_tensor


def build_stage1_train_transform(out_size):
    """
    阶段 1 训练用的 transform。
    当前第一版先和 eval 一样，只做基础 resize + tensor。
    """

    return BasicSegTransform(out_size)


def build_stage1_eval_transform(out_size):
    """
    阶段 1 验证用的 transform。
    当前第一版先只做基础 resize + tensor。
    """

    return BasicSegTransform(out_size)


def build_stage2_train_transform(image_size):
    """
    阶段 2 训练用的 transform。
    当前第一版先只做基础 resize + tensor。
    """

    return BasicSegTransform(image_size)


def build_stage2_eval_transform(image_size):
    """
    阶段 2 验证 / 推理用的 transform。
    当前第一版先只做基础 resize + tensor。
    """

    return BasicSegTransform(image_size)


def crop_patch(image, mask, x, y, crop_size, out_size):

    x = int(x)
    y = int(y)
    crop_size = int(crop_size)

    image_h, image_w = image.shape[:2]

    # 如果 x 太大，导致右边界超出图像，就把它拉回来
    x = max(0, min(x, image_w - crop_size))
    y = max(0, min(y, image_h - crop_size))

    # 用 numpy 切片裁 patch
    image_patch = image[y:y + crop_size, x:x + crop_size]
    mask_patch = mask[y:y + crop_size, x:x + crop_size]

    # 再把 patch resize 到统一大小
    image_patch = resize_image(image_patch, out_size)
    mask_patch = resize_mask(mask_patch, out_size)

    return image_patch, mask_patch


class ROIDataset(Dataset):
    #负责“整图分割阶段”的数据读取。
    def __init__(self, rows, image_size, transform=None, return_empty_mask_for_unlabeled=True):
        # 把输入 rows 变成列表，防止传进来的是别的可迭代对象
        self.rows = list(rows)

        self.image_size = image_size
        self.return_empty_mask_for_unlabeled = return_empty_mask_for_unlabeled

        # 如果外部没有传 transform，
        # 就默认使用阶段 2 的基础 eval transform
        if transform is None:
            self.transform = build_stage2_eval_transform(image_size)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        """
        这个函数的作用：
        取出第 index 条样本，并把它处理成模型可用的格式。
        """

        row = self.rows[index]

        # 1. 先读取彩色图
        image = read_image_rgb(row["image_path"])

        # 取原图尺寸
        original_h, original_w = image.shape[:2]

        # 2. 决定这条样本的 mask 怎么来
        mask_path = str(row.get("mask_path", "")).strip()

        # 如果有真实 mask 路径，就读取真实 mask
        if mask_path != "":
            mask = read_mask_binary(mask_path)

        else:
            # 如果没有真实 mask，但我们允许“自动生成空 mask”
            # 那就返回全零 mask
            if self.return_empty_mask_for_unlabeled:
                mask = build_empty_mask(original_h, original_w)
            else:
                raise ValueError(f"样本 {row['image_name']} 没有 mask_path，但当前设置不允许自动补空 mask")

        # 3. 做 transform
        image_tensor, mask_tensor = self.transform(image, mask)

        # 4. 整理成统一输出格式
        item = {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": row["sample_id"],
            "image_name": row["image_name"],
            "video_id": str(row.get("video_id", "")).strip(),
            "sample_type": row.get("sample_type", ""),
            "is_labeled": to_bool(row.get("is_labeled", False)),
        }

        return item
    
    
class PatchDataset(Dataset):
    """
    这个类的作用：
    负责“阶段 1 patch 训练”的数据读取。

    和 ROIDataset 的区别是：

    1. ROIDataset 处理的是整张图
    2. PatchDataset 处理的是“从整张图里裁出来的一小块 patch”

    PatchDataset 不直接接收整图 manifest，
    它接收的是“patch 索引表”的每一行。

    一条 patch_row 里，通常会包含这些信息：
        - image_path
        - mask_path
        - crop_x
        - crop_y
        - crop_size
        - out_size
        - patch_id
        - base_sample_id
        - patch_type
        - video_id
    """

    def __init__(self, patch_rows, transform=None):
        """
        参数：
            patch_rows:
                patch 索引行列表，也就是后面 build_patch_index.py 生成的 csv 读出来的内容

            transform:
                可选的预处理函数
                如果不传，我们就用 patch_row 里的 out_size 自动构建一个基础 transform
        """

        # 先把传进来的 patch_rows 变成列表，保证后面可以按下标访问
        self.patch_rows = list(patch_rows)

        # 保存 transform
        # 注意：
        # 这里允许 transform 为空，因为我们后面也可以根据每条 patch 的 out_size 自动处理
        self.transform = transform

    def __len__(self):
        """
        返回 patch 数据集里一共有多少条 patch。
        """

        return len(self.patch_rows)

    def __getitem__(self, index):
        """
        取出第 index 条 patch，并处理成模型可用格式。
        """

        # 取出当前这条 patch 的原始描述信息
        row = self.patch_rows[index]

        # ---------------------------
        # 第 1 步：读取原始整图
        # ---------------------------
        image = read_image_rgb(row["image_path"])

        # patch 阶段默认只会从 defect 图或 normal 图里裁 patch
        # 如果 mask_path 为空，说明这一条没有真实 mask
        # 那我们就给它构造一张全零 mask
        mask_path = str(row.get("mask_path", "")).strip()

        if mask_path != "":
            mask = read_mask_binary(mask_path)
        else:
            image_h, image_w = image.shape[:2]
            mask = build_empty_mask(image_h, image_w)

        # ---------------------------
        # 第 2 步：读取 patch 裁剪参数
        # ---------------------------
        # 注意：
        # csv 读出来以后，这些值通常都是字符串，
        # 所以这里要手动转成 int
        crop_x = int(row["crop_x"])
        crop_y = int(row["crop_y"])
        crop_size = int(row["crop_size"])
        out_size = int(row["out_size"])

        # ---------------------------
        # 第 3 步：从整图里裁出 patch
        # ---------------------------
        image_patch, mask_patch = crop_patch(
            image=image,
            mask=mask,
            x=crop_x,
            y=crop_y,
            crop_size=crop_size,
            out_size=out_size,
        )

        # ---------------------------
        # 第 4 步：做 transform
        # ---------------------------
        # 如果外部已经传了 transform，就用外部传入的
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_patch, mask_patch)

        else:
            # 如果外部没传，就根据这条 patch 的 out_size 自动建一个基础 transform
            patch_transform = build_stage1_eval_transform(out_size)
            image_tensor, mask_tensor = patch_transform(image_patch, mask_patch)

        # ---------------------------
        # 第 5 步：整理返回结果
        # ---------------------------
        item = {
            "image": image_tensor,
            "mask": mask_tensor,
            "patch_id": row.get("patch_id", ""),
            "base_sample_id": row.get("base_sample_id", ""),
            "patch_type": row.get("patch_type", ""),
            "video_id": str(row.get("video_id", "")).strip(),
        }

        return item
