"""
把原始数据分类， 扫描数据集，给每张图补上videoID， 交叉验证时进行划分，防止数据泄露
"""

import sys
from collections import Counter
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, read_csv_rows, save_json, write_csv_rows

#定义几个"全局路径常量"

DATASET_ROOT = PROJECT_ROOT / "dataset_new"
MANIFEST_DIR = PROJECT_ROOT/ "manifests"

def path_to_str(path: Path):
    return str(path.resolve())


def build_sample_id(prefix, image_name):
    """根据图片名称生成一个样本id 
    例如prefix=defect，images_name = 00131.jpg，输出defect_00131
    """

    stem = Path(image_name).stem #只保留名字去掉后缀
    return f"{prefix}_{stem}"

def scan_labeled_defects(root):
    #扫描已有的标注文件
    image_dir = root/"train"/"images" # / 表示拼路径
    mask_dir = root/"train"/"masks"

    rows = [] #返回列表

    for image_path in sorted(image_dir.glob("*.jpg")): #在images_dir这个文件夹里，找所有的.jpg文件，一个一个取出来，重新取stem
        stem = image_path.stem

        json_path = image_dir / f"{stem}.json"
        mask_path = mask_dir / f"{stem}.png"

        if not json_path.exists():
            continue

        if not mask_path.exists():
            continue

        #构建manifest记录
        row = {
            "sample_id": build_sample_id("defect",image_path.name),
            "image_name":image_path.name ,
            "image_path": path_to_str(image_path),
            "mask_path": path_to_str(mask_path),
            "json_path": path_to_str(json_path),
            "sample_type": "defect",
            "is_labeled": True,
            "source_split": "train",
        }

        rows.append(row)
    return rows

def scan_unlabeled_holdout(root):
    """
    扫描当前未标注的 holdout 图。
    """

    val_dir = root / "val"
    rows = []

    for image_path in sorted(val_dir.glob("*.jpg")):
        row = {
            "sample_id": build_sample_id("holdout", image_path.name),
            "image_name": image_path.name,
            "image_path": path_to_str(image_path),
            "mask_path": "",
            "json_path": "",
            "sample_type": "defect_holdout_unlabeled",
            "is_labeled": False,
            "source_split": "val",
        }

        rows.append(row)

    return rows


def scan_normals(root):
    """
    这个函数的作用：
    扫描正常图。

    路径是：
    - dataset_new/normal_crops_selected/*.jpg

    """

    normal_dir = root / "normal_crops_selected"
    rows = []

    for image_path in sorted(normal_dir.glob("*.jpg")):
        row = {
            "sample_id": build_sample_id("normal", image_path.name),
            "image_name": image_path.name,
            "image_path": path_to_str(image_path),
            "mask_path": "",
            "json_path": "",
            "sample_type": "normal",
            "is_labeled": False,
            "source_split": "normal_pool",
        }

        rows.append(row)

    return rows



MAPPING_DIR = DATASET_ROOT / "图片来源视频映射"

def load_mapping_by_image_name(path):

    """
    读取 defect 训练图的映射文件。

    返回一个以 image_name 为键的映射字典
    """
    rows = read_csv_rows(path)

    mapping = {}

    for row in rows:
        image_name = row.get("image_name","").strip()

        if image_name == "":
            continue
        mapping[image_name] = row #把当前这一行数据 row 存到字典 mapping 里，并且用 image_name 当键。


    return mapping


def load_defect_train_mapping(path=None):
    if path is None:
        path = MAPPING_DIR / "train_image_to_video.csv"
    return load_mapping_by_image_name(path)


def load_defect_holdout_mapping(path=None):
    if path is None:
        path = MAPPING_DIR / "val_image_to_video.csv"
    return load_mapping_by_image_name(path)

def load_normal_mapping(path=None):
    """
    读取正常图的映射文件。图片来自于哪一个视频
    """
    if path is None:
        path = DATASET_ROOT / "normal_crops_selected" / "video_mapping.csv"

    return load_mapping_by_image_name(path)

def attach_video_info(rows, mapping_dict):
    """
    把“样本列表”和“映射字典”对接起来。

        rows:前面 scan_xxx(...) 得到的样本列表

        mapping_dict: load_xxx_mapping(...) 得到的映射字典

    返回：
        1. attached_rows
           成功补上视频信息的样本列表
        2. missing_image_names
           没有在映射表里找到的图片名列表
    """

    # 用来保存“成功补上视频信息”的样本
    attached_rows = []

    # 用来记录“没找到映射”的图片名
    missing_image_names = []

    for row in rows:
        image_name = row["image_name"]

        # 去映射字典里查找这张图对应的信息
        mapping_row = mapping_dict.get(image_name)

        # 如果找不到，说明这张图没有映射信息
        # 那就先记下来，然后跳过
        if mapping_row is None:
            missing_image_names.append(image_name)
            continue

        # row.copy() 的意思是：
        # 复制一份原始字典
        # 这样我们是在“副本”上加字段，不会直接改原来的 row
        new_row = row.copy()

        # 补 video_id
        # str(...) 是为了保证写进 csv 时一定是字符串
        # .strip() 是为了去掉前后空格
        new_row["video_id"] = str(mapping_row.get("video_id", "")).strip()

      
        new_row["video_name"] = str(mapping_row.get("video_name", "")).strip()

        # 这里要特别注意：
        # normal 的映射文件字段叫 frame_id
        # train/val 的映射文件字段叫 source_frame_idx
        # 所以我们做一个“优先级处理”
        frame_id = mapping_row.get("frame_id", "")

        # 如果 frame_id 是空的，就再试一次 source_frame_idx
        if str(frame_id).strip() == "":
            frame_id = mapping_row.get("source_frame_idx", "")

        # 最后统一保存到 new_row["frame_id"]
        # 这样后面整个项目里就只认一个统一字段名：frame_id
        new_row["frame_id"] = str(frame_id).strip()

        # 到这里，这条样本就已经补完视频信息了
        attached_rows.append(new_row)

    return attached_rows, missing_image_names

#################################################################################################
# codex
def video_id_to_int(video_id):
    """
    这个函数的作用：
    把 video_id 从“字符串”转成“整数”。

    为什么要做这一步？

    因为 csv 读出来的内容，很多时候默认都是字符串。
    比如：
        "7"
        "16"

    如果直接按字符串排序，结果会变得不直观。
    例如字符串排序里，"16" 可能会排在 "7" 前面。

    所以我们统一先把 video_id 转成 int，
    这样排序时就会更符合人的直觉。
    """

    return int(str(video_id).strip())


def sort_video_id_list(video_id_list):
    """
    这个函数的作用：
    把一串 video_id 按“真正的数字大小”排序。

    参数：
        video_id_list:
            例如 ["16", "7", "30", "2"]

    返回：
        例如 ["2", "7", "16", "30"]
    """

    return sorted(video_id_list, key=video_id_to_int)


def build_master_manifest(defect_rows, holdout_rows, normal_rows):
    """
    这个函数的作用：
    把三类样本合并成一个总表 master manifest。

    你可以把它理解成：
    现在我们手里有三堆数据：
        1. defect
        2. holdout
        3. normal

    这个函数就是把它们拼成一张总表，
    方便后面统一查看和保存。

    参数：
        defect_rows:
            有标注缺陷图的样本列表

        holdout_rows:
            未标注 holdout 图的样本列表

        normal_rows:
            正常图的样本列表

    返回：
        一个总列表 all_rows
    """

    all_rows = []

    for row in defect_rows:
        all_rows.append(row)

    for row in holdout_rows:
        all_rows.append(row)

    for row in normal_rows:
        all_rows.append(row)

    return all_rows


def group_rows_by_video_id(rows):
    """
    这个函数的作用：
    按 video_id 对样本分组。

    原来 rows 是“很多样本混在一起的一个列表”。

    经过这个函数以后，会变成这样：

        {
            "2":  [属于 video 2 的所有样本],
            "7":  [属于 video 7 的所有样本],
            "16": [属于 video 16 的所有样本],
        }

    为什么必须分组？
    因为交叉验证不能按单张图片乱切，
    而是要按“整组视频”来切，防止同一个视频同时出现在训练和验证里。
    """

    grouped = {}

    for row in rows:
        video_id = row["video_id"]

        # 如果这个 video_id 还没在 grouped 里出现过，
        # 先给它创建一个空列表
        if video_id not in grouped:
            grouped[video_id] = []

        # 再把当前样本追加进去
        grouped[video_id].append(row)

    return grouped


def group_sample_count(group_info):
    """
    这个函数的作用：
    取出一个“视频组”的样本数量。

    这个函数主要是给 sorted(..., key=...) 用的。

    group_info 会长这样：
        {
            "video_id": "16",
            "rows": [...],
            "sample_count": 4
        }

    我们这里只返回 sample_count。
    """

    return group_info["sample_count"]


def choose_fold_with_smallest_sample_count(folds):
    """
    这个函数的作用：
    从多个 fold 里，选出“当前样本数最少”的那个 fold。

    为什么要这么做？
    因为我们希望 4 个 fold 尽量平衡，
    不要出现某一折特别大、另一折特别小。

    如果两个 fold 的 sample_count 一样，
    我们再继续比较：
        1. 谁的视频组更少
        2. 如果还一样，就选 fold_index 更小的那个

    这样做的好处是：
    整个分配过程更稳定，也更容易复现。
    """

    best_fold = None

    for fold in folds:
        # 第一个看到的 fold，先暂时当作当前最好选择
        if best_fold is None:
            best_fold = fold
            continue

        # 如果当前 fold 的样本数更少，就直接替换
        if fold["sample_count"] < best_fold["sample_count"]:
            best_fold = fold
            continue

        # 如果样本数相同，再比较视频组数量
        if fold["sample_count"] == best_fold["sample_count"]:
            if len(fold["video_ids"]) < len(best_fold["video_ids"]):
                best_fold = fold
                continue

            # 如果视频组数量也一样，再比较 fold_index
            if len(fold["video_ids"]) == len(best_fold["video_ids"]):
                if fold["fold_index"] < best_fold["fold_index"]:
                    best_fold = fold

    return best_fold


def build_defect_folds_by_video(defect_rows, n_folds=4, seed=42):
    """
    这个函数的作用：
    把 defect 样本按 video_id 分成 4 个 fold。

    注意：
    这里分的是“视频组”，不是“单张图片”。

    整体逻辑分 4 步：

    第 1 步：
        先按 video_id 把 defect_rows 分组

    第 2 步：
        把每个视频组整理成一个 group_info

    第 3 步：
        先固定随机种子打乱，再按 sample_count 从大到小排序
        这样可以让“大视频组”优先被分配，平衡效果更好

    第 4 步：
        每次都把当前视频组放进“样本数最少”的那个 fold

    参数：
        defect_rows:
            已经补好 video_id 的 defect 样本列表

        n_folds:
            折数
            你现在要做四折交叉验证，所以这里默认写 4

        seed:
            随机种子
            固定以后，每次划分结果都一致

    返回：
        folds:
            一个列表，长度等于 n_folds
            每个元素都是一个 fold 的信息字典
    """

    # 至少得 2 折才有意义
    if n_folds < 2:
        raise ValueError("n_folds 必须至少为 2")

    # 先按 video_id 分组
    grouped = group_rows_by_video_id(defect_rows)

    # 拿到所有不同的 video_id
    video_ids = list(grouped.keys())

    # 如果视频组数量比折数还少，就没法做交叉验证
    if len(video_ids) < n_folds:
        raise ValueError("视频组数量少于折数，无法做交叉验证")

    # 准备一个列表，里面每个元素都代表“一个视频组”
    video_groups = []

    for video_id in video_ids:
        video_rows = grouped[video_id]

        group_info = {
            "video_id": video_id,
            "rows": video_rows,
            "sample_count": len(video_rows),
        }

        video_groups.append(group_info)

    # 固定随机种子
    rng = random.Random(seed)

    # 先打乱一下，避免完全被原始顺序控制
    rng.shuffle(video_groups)

    # 再按“视频组的样本数”从大到小排序
    # 这样大的视频组会优先分配，更容易得到平衡的四折
    video_groups = sorted(video_groups, key=group_sample_count, reverse=True)

    # 创建空的 folds
    folds = []

    for fold_index in range(n_folds):
        fold_info = {
            "fold_index": fold_index,
            "video_ids": [],
            "rows": [],
            "sample_count": 0,
        }
        folds.append(fold_info)

    # 逐个分配视频组
    for group_info in video_groups:
        # 先选出“当前样本数最少”的那个 fold
        target_fold = choose_fold_with_smallest_sample_count(folds)

        # 把当前视频组的 video_id 记进去
        target_fold["video_ids"].append(group_info["video_id"])

        # 把当前视频组里的所有样本都加入这个 fold
        for row in group_info["rows"]:
            target_fold["rows"].append(row)

        # 更新这个 fold 的样本总数
        target_fold["sample_count"] += group_info["sample_count"]

    # 最后把每个 fold 里的 video_ids 排序一下，方便查看
    for fold in folds:
        fold["video_ids"] = sort_video_id_list(fold["video_ids"])

    return folds

# 这一组字段，是我们写 csv 时统一使用的表头顺序
# 为什么要专门写一个常量？
# 因为这样所有 manifest 的列顺序都会一致，后面读起来更稳定
MANIFEST_FIELDNAMES = [
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
]


def sort_rows_for_manifest(rows):
    """
    这个函数的作用：
    把样本列表按一个稳定的顺序排序。

    为什么要排序？
    因为如果不排序，每次写出来的 csv 行顺序可能会不太一样，
    这样你调试时会比较乱。

    这里我们主要按：
        1. video_id
        2. image_name
        3. sample_id
    来排序。
    """

    def sort_key(row):
        # 先取出 video_id
        video_id = str(row.get("video_id", "")).strip()

        # 如果 video_id 是空字符串，就给一个特殊值 -1
        # 正常情况下，经过 attach_video_info 以后，这里一般不会空
        if video_id == "":
            video_sort_value = -1
        else:
            video_sort_value = video_id_to_int(video_id)

        image_name = row.get("image_name", "")
        sample_id = row.get("sample_id", "")

        return (video_sort_value, image_name, sample_id)

    return sorted(rows, key=sort_key)


def collect_unique_video_ids(rows):
    """
    这个函数的作用：
    从一堆样本里，提取所有不重复的 video_id。

    例如：
        rows 里有 100 条样本
        但其实只来自 6 个视频

    那么这个函数就会返回这 6 个 video_id。
    """

    video_id_set = set()

    for row in rows:
        video_id = str(row["video_id"]).strip()
        video_id_set.add(video_id)

    return sort_video_id_list(list(video_id_set))


def count_duplicate_values(rows, key):
    counter = Counter(str(row.get(key, "")).strip() for row in rows)
    return sum(1 for value, count in counter.items() if value != "" and count > 1)


def build_video_distribution(rows):
    counter = Counter(str(row.get("video_id", "")).strip() for row in rows if str(row.get("video_id", "")).strip() != "")
    sorted_video_ids = sort_video_id_list(list(counter.keys()))
    return {video_id: int(counter[video_id]) for video_id in sorted_video_ids}


def build_data_audit(
    defect_rows_raw,
    holdout_rows_raw,
    normal_rows_raw,
    defect_rows,
    holdout_rows,
    normal_rows,
    defect_missing,
    holdout_missing,
    normal_missing,
):
    return {
        "raw_scan_counts": {
            "defect": len(defect_rows_raw),
            "holdout": len(holdout_rows_raw),
            "normal": len(normal_rows_raw),
        },
        "attached_counts": {
            "defect": len(defect_rows),
            "holdout": len(holdout_rows),
            "normal": len(normal_rows),
        },
        "unique_video_id_counts": {
            "defect": len(collect_unique_video_ids(defect_rows)),
            "holdout": len(collect_unique_video_ids(holdout_rows)),
            "normal": len(collect_unique_video_ids(normal_rows)),
        },
        "unique_image_name_counts": {
            "defect": len(set(row["image_name"] for row in defect_rows)),
            "holdout": len(set(row["image_name"] for row in holdout_rows)),
            "normal": len(set(row["image_name"] for row in normal_rows)),
        },
        "missing_mapping_counts": {
            "defect": len(defect_missing),
            "holdout": len(holdout_missing),
            "normal": len(normal_missing),
        },
        "missing_mapping_preview": {
            "defect": defect_missing[:10],
            "holdout": holdout_missing[:10],
            "normal": normal_missing[:10],
        },
        "duplicate_image_name_counts": {
            "defect": count_duplicate_values(defect_rows, "image_name"),
            "holdout": count_duplicate_values(holdout_rows, "image_name"),
            "normal": count_duplicate_values(normal_rows, "image_name"),
        },
        "duplicate_sample_id_counts": {
            "defect": count_duplicate_values(defect_rows, "sample_id"),
            "holdout": count_duplicate_values(holdout_rows, "sample_id"),
            "normal": count_duplicate_values(normal_rows, "sample_id"),
        },
        "per_video_sample_count": {
            "defect": build_video_distribution(defect_rows),
            "holdout": build_video_distribution(holdout_rows),
            "normal": build_video_distribution(normal_rows),
        },
    }


def build_train_val_rows_for_fold(folds, val_fold_index):
    """
    这个函数的作用：
    在已经分好的 4 个 fold 里，指定“哪一折当验证集”，
    然后自动得到：
        - defect_train_rows
        - defect_val_rows
        - defect_train_video_ids
        - defect_val_video_ids

    例如：
        如果 val_fold_index = 0
        那么第 0 折就是验证集
        其余第 1/2/3 折合起来就是训练集
    """

    # 先检查 fold 下标是否合法
    if val_fold_index < 0 or val_fold_index >= len(folds):
        raise ValueError("val_fold_index 超出了 fold 范围")

    train_rows = []
    val_rows = []

    train_video_ids = []
    val_video_ids = []

    # 遍历每一个 fold
    for fold in folds:
        # 如果当前 fold 恰好就是“本轮验证折”
        if fold["fold_index"] == val_fold_index:
            # 那它的所有样本都进入 val_rows
            for row in fold["rows"]:
                val_rows.append(row)

            # 它的视频 id 也进入 val_video_ids
            for video_id in fold["video_ids"]:
                val_video_ids.append(video_id)

        else:
            # 否则，这一折属于训练集
            for row in fold["rows"]:
                train_rows.append(row)

            for video_id in fold["video_ids"]:
                train_video_ids.append(video_id)

    # 去重并排序
    train_video_ids = sort_video_id_list(list(set(train_video_ids)))
    val_video_ids = sort_video_id_list(list(set(val_video_ids)))

    # 做一个非常重要的安全检查：
    # train 和 val 的 video_id 绝对不能有交集
    if not set(train_video_ids).isdisjoint(set(val_video_ids)):
        raise ValueError("发现 train 和 val 的 video_id 有重叠，说明切分出错了")

    # 返回时顺手把行顺序也排好
    train_rows = sort_rows_for_manifest(train_rows)
    val_rows = sort_rows_for_manifest(val_rows)

    return train_rows, val_rows, train_video_ids, val_video_ids


def split_normal_rows_for_fold(normal_rows, defect_val_video_ids, future_holdout_video_ids):
    """
    这个函数的作用：
    根据“当前这一折的 defect 验证视频”去切 normal 数据。

    切分规则是：

    1. 如果 normal 的 video_id 属于 future_holdout_video_ids
       -> 放进 normal_future_holdout

    2. 否则，如果 normal 的 video_id 属于 defect_val_video_ids
       -> 放进 normal_val

    3. 剩下的 normal
       -> 放进 normal_train

    为什么 future_holdout 优先级更高？
    因为这些视频我们想保留给未来 holdout 场景，不提前进入训练或当前验证。
    """

    defect_val_video_id_set = set(str(x).strip() for x in defect_val_video_ids)
    future_holdout_video_id_set = set(str(x).strip() for x in future_holdout_video_ids)

    normal_train_rows = []
    normal_val_rows = []
    normal_future_holdout_rows = []

    for row in normal_rows:
        video_id = str(row["video_id"]).strip()

        # 先判断是不是 future holdout 视频
        if video_id in future_holdout_video_id_set:
            normal_future_holdout_rows.append(row)
            continue

        # 再判断是不是当前这一折的 defect 验证视频
        if video_id in defect_val_video_id_set:
            normal_val_rows.append(row)
            continue

        # 剩下的都放 normal_train
        normal_train_rows.append(row)

    normal_train_rows = sort_rows_for_manifest(normal_train_rows)
    normal_val_rows = sort_rows_for_manifest(normal_val_rows)
    normal_future_holdout_rows = sort_rows_for_manifest(normal_future_holdout_rows)

    return normal_train_rows, normal_val_rows, normal_future_holdout_rows


def write_manifest_csv(path, rows):

    write_csv_rows(path, rows, MANIFEST_FIELDNAMES)


def raise_if_missing_mapping(mapping_name, missing_image_names):

    if len(missing_image_names) == 0:
        return

    preview = missing_image_names[:10]
    raise ValueError(
        f"{mapping_name} 有 {len(missing_image_names)} 张图片没找到映射，前 10 个是: {preview}"
    )


def main():
    """
    这是整个 prepare_dataset.py 的主函数。

    它会按下面顺序完成所有事情：

    1. 扫描 defect / holdout / normal
    2. 读取三个映射文件
    3. 给所有样本补上 video_id
    4. 写基础 manifest
    5. 构建 4 个 defect folds
    6. 为每一折生成 defect_train / defect_val
    7. 为每一折生成 normal_train / normal_val
    8. 写 split_summary.json
    """

    # 先确保 manifests 目录存在
    ensure_dir(MANIFEST_DIR)

    # ---------------------------
    # 第 1 步：扫描三类原始样本
    # ---------------------------
    defect_rows_raw = scan_labeled_defects(DATASET_ROOT)
    holdout_rows_raw = scan_unlabeled_holdout(DATASET_ROOT)
    normal_rows_raw = scan_normals(DATASET_ROOT)

    # ---------------------------
    # 第 2 步：读取三个映射文件
    # ---------------------------
    defect_train_mapping = load_defect_train_mapping()
    defect_holdout_mapping = load_defect_holdout_mapping()
    normal_mapping = load_normal_mapping()

    # ---------------------------
    # 第 3 步：给样本补 video 信息
    # ---------------------------
    defect_rows, defect_missing = attach_video_info(defect_rows_raw, defect_train_mapping)
    holdout_rows, holdout_missing = attach_video_info(holdout_rows_raw, defect_holdout_mapping)
    normal_rows, normal_missing = attach_video_info(normal_rows_raw, normal_mapping)

    # 如果有任何图片没找到映射，就直接报错
    raise_if_missing_mapping("defect_train_mapping", defect_missing)
    raise_if_missing_mapping("defect_holdout_mapping", holdout_missing)
    raise_if_missing_mapping("normal_mapping", normal_missing)

    data_audit = build_data_audit(
        defect_rows_raw=defect_rows_raw,
        holdout_rows_raw=holdout_rows_raw,
        normal_rows_raw=normal_rows_raw,
        defect_rows=defect_rows,
        holdout_rows=holdout_rows,
        normal_rows=normal_rows,
        defect_missing=defect_missing,
        holdout_missing=holdout_missing,
        normal_missing=normal_missing,
    )
    save_json(MANIFEST_DIR / "data_audit.json", data_audit)

    # ---------------------------
    # 第 4 步：写基础 manifest
    # ---------------------------
    master_manifest = build_master_manifest(defect_rows, holdout_rows, normal_rows)

    write_manifest_csv(MANIFEST_DIR / "master_manifest.csv", sort_rows_for_manifest(master_manifest))
    write_manifest_csv(MANIFEST_DIR / "defect_labeled.csv", sort_rows_for_manifest(defect_rows))
    write_manifest_csv(MANIFEST_DIR / "defect_holdout_unlabeled.csv", sort_rows_for_manifest(holdout_rows))
    write_manifest_csv(MANIFEST_DIR / "normal_pool.csv", sort_rows_for_manifest(normal_rows))

    # ---------------------------
    # 第 5 步：准备 future holdout 视频集合
    # ---------------------------
    # 这里的 holdout 指的是 dataset_new/val 里的未标注图
    # 它们对应的视频，未来希望保留为更独立的评估域
    future_holdout_video_ids = collect_unique_video_ids(holdout_rows)

    # 先单独生成一次 normal_future_holdout.csv
    # 注意这里 defect_val_video_ids 传空列表，
    # 因为我们这一步只是想先把“未来 holdout 的 normal 视频”挑出来
    _, _, normal_future_holdout_rows = split_normal_rows_for_fold(
        normal_rows=normal_rows,
        defect_val_video_ids=[],
        future_holdout_video_ids=future_holdout_video_ids,
    )

    write_manifest_csv(
        MANIFEST_DIR / "normal_future_holdout.csv",
        normal_future_holdout_rows,
    )

    # ---------------------------
    # 第 6 步：构建 4 个 defect folds
    # ---------------------------
    n_folds = 4
    seed = 42

    folds = build_defect_folds_by_video(defect_rows, n_folds=n_folds, seed=seed)

    # ---------------------------
    # 第 7 步：为每一折生成 train / val 清单
    # ---------------------------
    summary = {
        "n_folds": n_folds,
        "seed": seed,
        "master_manifest_count": len(master_manifest),
        "defect_labeled_count": len(defect_rows),
        "defect_holdout_unlabeled_count": len(holdout_rows),
        "normal_pool_count": len(normal_rows),
        "normal_future_holdout_count": len(normal_future_holdout_rows),
        "future_holdout_video_ids": future_holdout_video_ids,
        "folds": [],
    }

    for fold_index in range(n_folds):
        # 先得到当前这一折的 defect train / val
        defect_train_rows, defect_val_rows, defect_train_video_ids, defect_val_video_ids = build_train_val_rows_for_fold(
            folds=folds,
            val_fold_index=fold_index,
        )

        # 再根据“当前这一折的 defect_val_video_ids”去切 normal
        normal_train_rows, normal_val_rows, _ = split_normal_rows_for_fold(
            normal_rows=normal_rows,
            defect_val_video_ids=defect_val_video_ids,
            future_holdout_video_ids=future_holdout_video_ids,
        )

        # 写出当前 fold 的 defect manifest
        write_manifest_csv(
            MANIFEST_DIR / f"defect_fold{fold_index}_train.csv",
            defect_train_rows,
        )
        write_manifest_csv(
            MANIFEST_DIR / f"defect_fold{fold_index}_val.csv",
            defect_val_rows,
        )

        # 写出当前 fold 的 normal manifest
        write_manifest_csv(
            MANIFEST_DIR / f"normal_fold{fold_index}_train.csv",
            normal_train_rows,
        )
        write_manifest_csv(
            MANIFEST_DIR / f"normal_fold{fold_index}_val.csv",
            normal_val_rows,
        )

        # 把这一折的统计信息写进 summary
        fold_summary = {
            "fold_index": fold_index,
            "defect_train_count": len(defect_train_rows),
            "defect_val_count": len(defect_val_rows),
            "normal_train_count": len(normal_train_rows),
            "normal_val_count": len(normal_val_rows),
            "defect_train_video_ids": defect_train_video_ids,
            "defect_val_video_ids": defect_val_video_ids,
        }

        summary["folds"].append(fold_summary)

        # 打印一行简短日志，方便你在终端里看
        print(
            f"fold {fold_index}: "
            f"defect_train={len(defect_train_rows)}, "
            f"defect_val={len(defect_val_rows)}, "
            f"normal_train={len(normal_train_rows)}, "
            f"normal_val={len(normal_val_rows)}"
        )

    # ---------------------------
    # 第 8 步：保存 summary JSON
    # ---------------------------
    save_json(MANIFEST_DIR / "split_summary.json", summary)

    print("prepare_dataset 完成，所有 manifest 已写入 manifests/ 目录。")


if __name__ == "__main__":
    main()
