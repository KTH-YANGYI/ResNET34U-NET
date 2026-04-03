import random
import sys
from PIL import Image

# Path 用来处理文件路径
from pathlib import Path

# numpy 用来处理 mask 坐标
import numpy as np


# 这三行的作用和你在 prepare_dataset.py 里写的是一样的：
# 找到项目根目录，并把它加入 Python 的搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 从我们已经写好的模块里导入工具函数
from src.datasets import read_mask_binary
from src.utils import ensure_dir, read_csv_rows, save_json, write_csv_rows


# 这里定义几个路径常量，后面会一直用到
MANIFEST_DIR = PROJECT_ROOT / "manifests"


def mask_to_bbox(mask):
    """
    这个函数的作用：
    根据二值 mask 计算最小外接框 bbox。

    输入：
        mask 是一个二维数组，形状通常是 H x W
        里面 0 表示背景，1 表示缺陷

    输出：
        如果 mask 里有前景，返回一个字典，例如：
        {
            "x_min": 100,
            "y_min": 200,
            "x_max": 150,
            "y_max": 230,
            "bbox_w": 51,
            "bbox_h": 31,
            "center_x": 125,
            "center_y": 215,
        }

        如果 mask 里完全没有前景，就返回 None
    """

    # np.where(mask > 0) 的意思是：
    # 找出所有前景像素的位置
    ys, xs = np.where(mask > 0)

    # 如果一个前景像素都没有，说明这张 mask 是空的
    if len(xs) == 0:
        return None

    # 最小 x / 最大 x
    x_min = int(xs.min())
    x_max = int(xs.max())

    # 最小 y / 最大 y
    y_min = int(ys.min())
    y_max = int(ys.max())

    # 外接框宽高
    # 这里要 +1，因为坐标是“包含两端”的
    bbox_w = x_max - x_min + 1
    bbox_h = y_max - y_min + 1

    # 中心点坐标
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "center_x": center_x,
        "center_y": center_y,
    }


def clip_crop_window(center_x, center_y, crop_size, image_w, image_h):
    """
    这个函数的作用：
    给定“希望围绕哪个中心裁 patch”，
    计算 patch 左上角坐标，并保证 patch 不会跑出图像边界。

    输入：
        center_x, center_y:
            patch 想围绕的中心位置

        crop_size:
            要裁的正方形边长

        image_w, image_h:
            原图宽和高

    输出：
        crop_x, crop_y
        也就是 patch 左上角坐标
    """

    crop_size = int(crop_size)

    # 如果 crop_size 比原图还大，那一定没法裁
    if crop_size > image_w or crop_size > image_h:
        raise ValueError("crop_size 不能大于原图尺寸")

    # 希望 patch 的中心对齐到 center_x / center_y
    # 所以左上角大致是 中心 - 半边长
    crop_x = int(round(center_x - crop_size / 2.0))
    crop_y = int(round(center_y - crop_size / 2.0))

    # 下面这两行是在做“边界裁剪”
    # 如果左上角太靠左，就拉回到 0
    # 如果右下角会越界，就把左上角往回拉
    crop_x = max(0, min(crop_x, image_w - crop_size))
    crop_y = max(0, min(crop_y, image_h - crop_size))

    return crop_x, crop_y


def sample_crop_window_containing_bbox(bbox, crop_size, image_w, image_h, rng):
    """
    这个函数的作用：
    随机采样一个 patch 窗口，但要求：
    “这个 patch 必须把整个缺陷 bbox 完整包住”。

    这个函数后面会被 positive_shift 和 positive_context 共用。

    为什么它重要？
    因为我们希望正样本 patch 里，裂纹不能被裁断。

    返回：
        如果能找到合法窗口，返回 (crop_x, crop_y)
        如果找不到，返回 None
    """

    crop_size = int(crop_size)

    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]

    # 一个 patch 要完整包含 bbox，
    # 那么 patch 左上角 crop_x 的可选范围必须满足：
    # 1. crop_x <= x_min
    # 2. crop_x + crop_size - 1 >= x_max
    #
    # 推出来就是：
    # x_max - crop_size + 1 <= crop_x <= x_min
    #
    # 同时还要满足 patch 不越界：
    # 0 <= crop_x <= image_w - crop_size
    x_low = max(0, x_max - crop_size + 1)
    x_high = min(x_min, image_w - crop_size)

    # y 方向同理
    y_low = max(0, y_max - crop_size + 1)
    y_high = min(y_min, image_h - crop_size)

    # 如果下界比上界还大，说明根本不存在合法窗口
    if x_low > x_high or y_low > y_high:
        return None

    # 在合法范围里随机采一个位置
    crop_x = rng.randint(int(x_low), int(x_high))
    crop_y = rng.randint(int(y_low), int(y_high))

    return crop_x, crop_y


def build_patch_row(row, patch_index, patch_type, crop_x, crop_y, crop_size, out_size):
    """
    这个函数的作用：
    把“一块 patch 的描述信息”整理成一条字典记录。

    这条记录后面会写进 patch index csv 里。

    参数：
        row:
            原始 defect manifest 里的一条样本记录

        patch_index:
            这是这张图生成的第几个 patch

        patch_type:
            patch 类型，例如：
                positive_center
                positive_shift
                positive_context

        crop_x, crop_y:
            patch 左上角坐标

        crop_size:
            原始裁剪边长

        out_size:
            裁完以后，后面会 resize 到多大
    """

    patch_id = f"{row['sample_id']}_{patch_type}_{patch_index:02d}"

    patch_row = {
        "patch_id": patch_id,
        "base_sample_id": row["sample_id"],
        "image_path": row["image_path"],
        "mask_path": row["mask_path"],
        "patch_type": patch_type,
        "video_id": row["video_id"],
        "crop_x": int(crop_x),
        "crop_y": int(crop_y),
        "crop_size": int(crop_size),
        "out_size": int(out_size),
    }

    return patch_row


def make_window_key(crop_x, crop_y, crop_size):
    """
    这个函数的作用：
    把一个 patch 窗口唯一表示成一个三元组。

    只要这三个值完全一样，
    我们就认为这是“同一个 patch 窗口”：
        - crop_x
        - crop_y
        - crop_size
    """

    return (int(crop_x), int(crop_y), int(crop_size))


def add_patch_row_if_new(
    patch_rows,
    used_window_keys,
    row,
    patch_index,
    patch_type,
    crop_x,
    crop_y,
    crop_size,
    out_size,
):
    """
    这个函数的作用：
    只有当这个 patch 窗口以前没出现过时，才把它加入 patch_rows。

    返回值：
        True  表示这条 patch 成功加入了
        False 表示这条 patch 和已有窗口重复了，所以被跳过
    """

    window_key = make_window_key(crop_x, crop_y, crop_size)

    # 如果这个窗口已经出现过，就不再重复加入
    if window_key in used_window_keys:
        return False

    patch_row = build_patch_row(
        row=row,
        patch_index=patch_index,
        patch_type=patch_type,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_size=crop_size,
        out_size=out_size,
    )

    patch_rows.append(patch_row)
    used_window_keys.add(window_key)

    return True


def make_positive_center_patches(
    row,
    mask,
    rng,
    crop_sizes=(320, 384, 448),
    out_size=384,
    num_patches=1,
):
    """
    这个函数的作用：
    生成“中心型正样本 patch”。

    这里我们现在只保留 1 个，
    因为中心型 patch 太容易重复了。
    """

    bbox = mask_to_bbox(mask)

    if bbox is None:
        return []

    image_h, image_w = mask.shape[:2]

    patch_rows = []
    used_window_keys = set()

    # 这里虽然 num_patches 默认是 1，
    # 但仍然保留循环写法，这样以后你想改数量也方便
    for patch_index in range(num_patches):
        crop_size = rng.choice(list(crop_sizes))

        crop_x, crop_y = clip_crop_window(
            center_x=bbox["center_x"],
            center_y=bbox["center_y"],
            crop_size=crop_size,
            image_w=image_w,
            image_h=image_h,
        )

        add_patch_row_if_new(
            patch_rows=patch_rows,
            used_window_keys=used_window_keys,
            row=row,
            patch_index=patch_index,
            patch_type="positive_center",
            crop_x=crop_x,
            crop_y=crop_y,
            crop_size=crop_size,
            out_size=out_size,
        )

    return patch_rows


def make_positive_shift_patches(
    row,
    mask,
    rng,
    crop_sizes=(320, 384, 448),
    out_size=384,
    num_patches=2,
):
    """
    这个函数的作用：
    生成“平移型正样本 patch”。

    这里保留 2 个，因为它们比中心 patch 更有多样性。

    同时我们加入“去重”逻辑：
    如果随机到和已有 patch 完全一样的窗口，就继续重试。
    """

    bbox = mask_to_bbox(mask)

    if bbox is None:
        return []

    image_h, image_w = mask.shape[:2]

    patch_rows = []
    used_window_keys = set()

    # 为了防止随机重试太多次，这里设置一个最大尝试次数
    max_attempts = num_patches * 10
    attempt_count = 0
    patch_index = 0

    while len(patch_rows) < num_patches and attempt_count < max_attempts:
        attempt_count += 1

        crop_size = rng.choice(list(crop_sizes))

        sampled_window = sample_crop_window_containing_bbox(
            bbox=bbox,
            crop_size=crop_size,
            image_w=image_w,
            image_h=image_h,
            rng=rng,
        )

        if sampled_window is None:
            crop_x, crop_y = clip_crop_window(
                center_x=bbox["center_x"],
                center_y=bbox["center_y"],
                crop_size=crop_size,
                image_w=image_w,
                image_h=image_h,
            )
        else:
            crop_x, crop_y = sampled_window

        added = add_patch_row_if_new(
            patch_rows=patch_rows,
            used_window_keys=used_window_keys,
            row=row,
            patch_index=patch_index,
            patch_type="positive_shift",
            crop_x=crop_x,
            crop_y=crop_y,
            crop_size=crop_size,
            out_size=out_size,
        )

        # 只有成功加入时，patch_index 才往后走
        if added:
            patch_index += 1

    return patch_rows


def make_positive_context_patches(
    row,
    mask,
    rng,
    crop_sizes=(384, 448),
    out_size=384,
    num_patches=1,
):
    """
    这个函数的作用：
    生成“上下文型正样本 patch”。

    这里也只保留 1 个。
    同时让它更偏向较大的 crop_size，
    这样更容易保留裂纹周围环境。
    """

    bbox = mask_to_bbox(mask)

    if bbox is None:
        return []

    image_h, image_w = mask.shape[:2]

    patch_rows = []
    used_window_keys = set()

    max_attempts = num_patches * 10
    attempt_count = 0
    patch_index = 0

    while len(patch_rows) < num_patches and attempt_count < max_attempts:
        attempt_count += 1

        crop_size = rng.choice(list(crop_sizes))

        sampled_window = sample_crop_window_containing_bbox(
            bbox=bbox,
            crop_size=crop_size,
            image_w=image_w,
            image_h=image_h,
            rng=rng,
        )

        if sampled_window is None:
            crop_x, crop_y = clip_crop_window(
                center_x=bbox["center_x"],
                center_y=bbox["center_y"],
                crop_size=crop_size,
                image_w=image_w,
                image_h=image_h,
            )
        else:
            crop_x, crop_y = sampled_window

        added = add_patch_row_if_new(
            patch_rows=patch_rows,
            used_window_keys=used_window_keys,
            row=row,
            patch_index=patch_index,
            patch_type="positive_context",
            crop_x=crop_x,
            crop_y=crop_y,
            crop_size=crop_size,
            out_size=out_size,
        )

        if added:
            patch_index += 1

    return patch_rows


def get_image_hw(image_path):
    """
    这个函数的作用：
    读取一张图片的尺寸，但不把整张图真正读成 numpy 数组。

    为什么这么做？
    因为我们现在只想知道这张图的高和宽，
    不想为了一个尺寸就把整张图完整加载进内存。

    PIL 里：
        image.size 返回的是 (width, height)

    但我们后面大多数代码习惯写成：
        (height, width)

    所以这里我们会手动调整顺序。
    """

    image_path = Path(image_path)

    with Image.open(image_path) as image:
        image_w, image_h = image.size

    return image_h, image_w


def crop_window_to_box(crop_x, crop_y, crop_size):
    """
    这个函数的作用：
    把 patch 窗口转换成一个矩形框表示。

    输入：
        crop_x, crop_y:
            patch 左上角坐标

        crop_size:
            patch 边长

    输出：
        一个字典，表示这个 patch 对应的矩形框：
        {
            "x_min": ...,
            "y_min": ...,
            "x_max": ...,
            "y_max": ...,
        }
    """

    crop_x = int(crop_x)
    crop_y = int(crop_y)
    crop_size = int(crop_size)

    return {
        "x_min": crop_x,
        "y_min": crop_y,
        "x_max": crop_x + crop_size - 1,
        "y_max": crop_y + crop_size - 1,
    }


def boxes_intersect(box_a, box_b):
    """
    这个函数的作用：
    判断两个矩形框是否有重叠。

    返回：
        True  表示相交
        False 表示不相交
    """

    # 如果 A 在 B 的左边，且完全分开
    if box_a["x_max"] < box_b["x_min"]:
        return False

    # 如果 A 在 B 的右边，且完全分开
    if box_a["x_min"] > box_b["x_max"]:
        return False

    # 如果 A 在 B 的上方，且完全分开
    if box_a["y_max"] < box_b["y_min"]:
        return False

    # 如果 A 在 B 的下方，且完全分开
    if box_a["y_min"] > box_b["y_max"]:
        return False

    # 否则说明它们有相交
    return True


def expand_bbox(bbox, margin, image_w, image_h):
    """
    这个函数的作用：
    在原始 defect bbox 外面再扩一圈“安全边”。

    为什么要扩？
    因为 hard negative 不只是不能直接压到缺陷像素，
    最好也不要离缺陷太近。
    否则它虽然 technically 没裁到正像素，
    但可能仍然太靠近裂纹边缘，容易让负样本定义变得含糊。

    参数：
        bbox:
            原始缺陷框

        margin:
            向外扩多少像素

        image_w, image_h:
            原图宽高
            因为扩完以后仍然不能越界

    返回：
        一个扩张后的 bbox 字典
    """

    x_min = max(0, int(bbox["x_min"]) - int(margin))
    y_min = max(0, int(bbox["y_min"]) - int(margin))
    x_max = min(int(image_w) - 1, int(bbox["x_max"]) + int(margin))
    y_max = min(int(image_h) - 1, int(bbox["y_max"]) + int(margin))

    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
    }


def window_has_positive(mask, crop_x, crop_y, crop_size):
    """
    这个函数的作用：
    判断一个 patch 窗口里面是否含有任何正像素。

    返回：
        True  表示 patch 内有缺陷像素
        False 表示 patch 内完全没有缺陷像素
    """

    crop_x = int(crop_x)
    crop_y = int(crop_y)
    crop_size = int(crop_size)

    patch_mask = mask[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

    return bool((patch_mask > 0).any())


def make_hard_negative_patches(
    row,
    mask,
    rng,
    crop_sizes=(320, 384, 448),
    out_size=384,
    num_patches=4,
    safety_margin=24,
):
    """
    这个函数的作用：
    从 defect 图里生成 hard negative patch。

    规则：
        1. patch 内不能有任何正像素
        2. patch 还要和缺陷 bbox 保持一定安全边
        3. 每张 defect 图默认生成 4 个

    为什么 hard negative 有意义？
    因为这些 patch 来自“有缺陷图的背景区域”，
    它们比 normal 图更难，
    能帮助模型学会不要把一些复杂背景误判成裂纹。
    """

    bbox = mask_to_bbox(mask)

    # 如果这张 defect 图的 mask 是空的，就没法生成 hard negative
    if bbox is None:
        return []

    image_h, image_w = mask.shape[:2]

    # 先把缺陷框向外扩一圈，形成“禁区”
    expanded_bbox = expand_bbox(
        bbox=bbox,
        margin=safety_margin,
        image_w=image_w,
        image_h=image_h,
    )

    patch_rows = []
    used_window_keys = set()

    # 随机采样有可能失败，所以给足够多重试机会
    max_attempts = num_patches * 50
    attempt_count = 0
    patch_index = 0

    while len(patch_rows) < num_patches and attempt_count < max_attempts:
        attempt_count += 1

        crop_size = rng.choice(list(crop_sizes))

        # 如果 crop_size 比原图还大，就跳过
        if crop_size > image_w or crop_size > image_h:
            continue

        # 在整张图内随机采一个 patch 左上角
        crop_x = rng.randint(0, image_w - crop_size)
        crop_y = rng.randint(0, image_h - crop_size)

        # 先检查 patch 里面有没有正像素
        if window_has_positive(mask, crop_x, crop_y, crop_size):
            continue

        # 再检查这个 patch 会不会碰到“扩张后的缺陷禁区”
        patch_box = crop_window_to_box(crop_x, crop_y, crop_size)

        if boxes_intersect(patch_box, expanded_bbox):
            continue

        added = add_patch_row_if_new(
            patch_rows=patch_rows,
            used_window_keys=used_window_keys,
            row=row,
            patch_index=patch_index,
            patch_type="hard_negative",
            crop_x=crop_x,
            crop_y=crop_y,
            crop_size=crop_size,
            out_size=out_size,
        )

        if added:
            patch_index += 1

    return patch_rows


def make_normal_negative_patches(
    normal_rows,
    target_count,
    rng,
    crop_sizes=(320, 384, 448),
    out_size=384,
):
    """
    这个函数的作用：
    从 normal 图里随机采样 normal negative patch。

    参数：
        normal_rows:
            normal manifest 里的样本列表

        target_count:
            希望总共生成多少个 normal negative patch

    设计思想：
        我们希望 normal negative 的总量大致接近 hard negative 的总量，
        不要让 normal 类负样本一口气压倒所有别的 patch。

    实现方式：
        1. 先随机打乱 normal_rows
        2. 优先做到“一张 normal 图最多先取一个 patch”
        3. 如果 target_count 比 normal_rows 还多，再允许重复使用图片
    """

    if target_count <= 0:
        return []

    normal_rows = list(normal_rows)

    if len(normal_rows) == 0:
        return []

    shuffled_rows = normal_rows[:]
    rng.shuffle(shuffled_rows)

    selected_rows = []

    # 第一轮：先尽量不重复图片
    for row in shuffled_rows:
        if len(selected_rows) >= target_count:
            break
        selected_rows.append(row)

    # 如果 normal 图数量仍然不够，就允许重复采样
    while len(selected_rows) < target_count:
        selected_rows.append(rng.choice(normal_rows))

    patch_rows = []

    # 用一个字典来记录：
    # 每张 normal 图已经用过哪些窗口，避免完全重复
    used_window_keys_by_image = {}

    for patch_index, row in enumerate(selected_rows):
        image_path = row["image_path"]

        image_h, image_w = get_image_hw(image_path)

        if image_path not in used_window_keys_by_image:
            used_window_keys_by_image[image_path] = set()

        used_window_keys = used_window_keys_by_image[image_path]

        max_attempts = 20
        attempt_count = 0
        added = False

        while not added and attempt_count < max_attempts:
            attempt_count += 1

            crop_size = rng.choice(list(crop_sizes))

            if crop_size > image_w or crop_size > image_h:
                continue

            crop_x = rng.randint(0, image_w - crop_size)
            crop_y = rng.randint(0, image_h - crop_size)

            added = add_patch_row_if_new(
                patch_rows=patch_rows,
                used_window_keys=used_window_keys,
                row=row,
                patch_index=patch_index,
                patch_type="normal_negative",
                crop_x=crop_x,
                crop_y=crop_y,
                crop_size=crop_size,
                out_size=out_size,
            )

        # 如果重试很多次还是失败，
        # 就跳过这张图，不强行报错
        # 因为这只是第一版实现，先保证流程能跑
        # 后面如果真的发现数量不足，再专门优化
        if not added:
            continue

    return patch_rows

# 这是 patch index csv 的统一表头
# 后面所有阶段 1 patch 索引文件都会按这个顺序写
PATCH_INDEX_FIELDNAMES = [
    "patch_id",
    "base_sample_id",
    "image_path",
    "mask_path",
    "patch_type",
    "video_id",
    "crop_x",
    "crop_y",
    "crop_size",
    "out_size",
]


def sort_patch_rows(patch_rows):
    """
    这个函数的作用：
    把 patch_rows 按稳定顺序排序。

    为什么要排序？
    因为如果不排序，每次写出来的 csv 行顺序可能会乱一点，
    你之后调试会比较难看。

    这里我们主要按：
        1. base_sample_id
        2. patch_type
        3. patch_id
    排序。
    """

    def sort_key(row):
        return (
            row.get("base_sample_id", ""),
            row.get("patch_type", ""),
            row.get("patch_id", ""),
        )

    return sorted(patch_rows, key=sort_key)


def build_positive_patch_rows_for_defect(row, mask, rng, out_size=384):
    """
    这个函数的作用：
    给一张 defect 图生成全部正样本 patch。

    当前规则是：
        - positive_center: 1 个
        - positive_shift: 2 个
        - positive_context: 1 个

    所以理想情况下，一张 defect 图会得到 4 个正样本 patch。
    """

    patch_rows = []

    patch_rows.extend(
        make_positive_center_patches(
            row=row,
            mask=mask,
            rng=rng,
            out_size=out_size,
            num_patches=1,
        )
    )

    patch_rows.extend(
        make_positive_shift_patches(
            row=row,
            mask=mask,
            rng=rng,
            out_size=out_size,
            num_patches=2,
        )
    )

    patch_rows.extend(
        make_positive_context_patches(
            row=row,
            mask=mask,
            rng=rng,
            out_size=out_size,
            num_patches=1,
        )
    )

    return patch_rows


def build_patch_index_for_split(defect_rows, normal_rows, seed, out_size=384):
    """
    这个函数的作用：
    给“某一个数据划分”生成完整 patch index。

    这里的“某一个数据划分”可以是：
        - 训练集
        - 验证集

    它会生成三类 patch：
        1. 正样本 patch
        2. hard_negative patch
        3. normal_negative patch

    返回：
        1. all_patch_rows
           全部 patch 记录列表

        2. summary
           一个统计字典，方便后面写到 summary json
    """

    # 固定随机种子，保证可复现
    rng = random.Random(seed)

    positive_patch_rows = []
    hard_negative_patch_rows = []

    # 逐张 defect 图处理
    for row in defect_rows:
        # 读取这张图的真实 mask
        mask = read_mask_binary(row["mask_path"])

        # 生成正样本 patch
        positive_patch_rows.extend(
            build_positive_patch_rows_for_defect(
                row=row,
                mask=mask,
                rng=rng,
                out_size=out_size,
            )
        )

        # 生成 hard negative patch
        hard_negative_patch_rows.extend(
            make_hard_negative_patches(
                row=row,
                mask=mask,
                rng=rng,
                out_size=out_size,
                num_patches=4,
            )
        )

    # normal negative 的总数，和 hard negative 的总数对齐
    # 这样负样本里“来自 defect 背景”和“来自纯 normal 图”的比例更平衡
    normal_negative_patch_rows = make_normal_negative_patches(
        normal_rows=normal_rows,
        target_count=len(hard_negative_patch_rows),
        rng=rng,
        out_size=out_size,
    )

    # 把三类 patch 合并
    all_patch_rows = []

    for row in positive_patch_rows:
        all_patch_rows.append(row)

    for row in hard_negative_patch_rows:
        all_patch_rows.append(row)

    for row in normal_negative_patch_rows:
        all_patch_rows.append(row)

    # 排序，方便检查
    all_patch_rows = sort_patch_rows(all_patch_rows)

    # 做一个统计摘要
    summary = {
        "positive_count": len(positive_patch_rows),
        "hard_negative_count": len(hard_negative_patch_rows),
        "normal_negative_count": len(normal_negative_patch_rows),
        "total_count": len(all_patch_rows),
    }

    return all_patch_rows, summary


def build_train_patch_index(defect_train_rows, normal_train_rows, seed, out_size=384):
    """
    这个函数的作用：
    为某一折的训练集生成 patch index。
    """

    return build_patch_index_for_split(
        defect_rows=defect_train_rows,
        normal_rows=normal_train_rows,
        seed=seed,
        out_size=out_size,
    )


def build_val_patch_index(defect_val_rows, normal_val_rows, seed, out_size=384):
    """
    这个函数的作用：
    为某一折的验证集生成 patch index。
    """

    return build_patch_index_for_split(
        defect_rows=defect_val_rows,
        normal_rows=normal_val_rows,
        seed=seed,
        out_size=out_size,
    )


def write_patch_index_csv(path, patch_rows):
    """
    这个函数的作用：
    把 patch_rows 写成 csv 文件。
    """

    path = Path(path)

    # 确保父目录存在
    ensure_dir(path.parent)

    write_csv_rows(path, patch_rows, PATCH_INDEX_FIELDNAMES)


def main():
    """
    这是整个 build_patch_index.py 的主函数。

    它会做这些事：

    1. 读取四折的 defect / normal manifest
    2. 为每一折生成 train patch index
    3. 为每一折生成 val patch index
    4. 把结果写入 manifests/
    5. 保存一个 summary json
    """

    n_folds = 4
    patch_out_size = 384

    summary = {
        "n_folds": n_folds,
        "patch_out_size": patch_out_size,
        "folds": [],
    }

    for fold_index in range(n_folds):
        # ---------------------------
        # 读取当前折的 manifest
        # ---------------------------
        defect_train_rows = read_csv_rows(MANIFEST_DIR / f"defect_fold{fold_index}_train.csv")
        defect_val_rows = read_csv_rows(MANIFEST_DIR / f"defect_fold{fold_index}_val.csv")
        normal_train_rows = read_csv_rows(MANIFEST_DIR / f"normal_fold{fold_index}_train.csv")
        normal_val_rows = read_csv_rows(MANIFEST_DIR / f"normal_fold{fold_index}_val.csv")

        # 训练集和验证集用不同 seed
        # 这样能保证两边 patch 采样不会完全耦合在一起
        train_seed = 1000 + fold_index
        val_seed = 2000 + fold_index

        # ---------------------------
        # 生成 train patch index
        # ---------------------------
        train_patch_rows, train_summary = build_train_patch_index(
            defect_train_rows=defect_train_rows,
            normal_train_rows=normal_train_rows,
            seed=train_seed,
            out_size=patch_out_size,
        )

        # ---------------------------
        # 生成 val patch index
        # ---------------------------
        val_patch_rows, val_summary = build_val_patch_index(
            defect_val_rows=defect_val_rows,
            normal_val_rows=normal_val_rows,
            seed=val_seed,
            out_size=patch_out_size,
        )

        # ---------------------------
        # 写 csv 文件
        # ---------------------------
        train_index_path = MANIFEST_DIR / f"stage1_fold{fold_index}_train_index.csv"
        val_index_path = MANIFEST_DIR / f"stage1_fold{fold_index}_val_index.csv"

        write_patch_index_csv(train_index_path, train_patch_rows)
        write_patch_index_csv(val_index_path, val_patch_rows)

        # ---------------------------
        # 记录这一折的 summary
        # ---------------------------
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

        # 在终端里打印一行简短日志
        print(
            f"fold {fold_index}: "
            f"train_total={train_summary['total_count']} "
            f"(pos={train_summary['positive_count']}, "
            f"hard_neg={train_summary['hard_negative_count']}, "
            f"normal_neg={train_summary['normal_negative_count']}), "
            f"val_total={val_summary['total_count']} "
            f"(pos={val_summary['positive_count']}, "
            f"hard_neg={val_summary['hard_negative_count']}, "
            f"normal_neg={val_summary['normal_negative_count']})"
        )

    # 保存总 summary
    save_json(MANIFEST_DIR / "stage1_patch_summary.json", summary)

    print("build_patch_index 完成，所有 stage1 patch index 已写入 manifests/ 目录。")


if __name__ == "__main__":
    main()
