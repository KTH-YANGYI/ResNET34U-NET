# U-Net 两阶段裂纹分割项目手写实现计划

## 1. 项目目标

这个项目的目标不是一步做到最终论文版，而是先手写出一个可运行、可验证、可扩展的 U-Net 两阶段分割基线。

当前版本的目标固定为：

1. 读取现有 ROI 图像、mask 和标注信息；
2. 先做局部 patch 训练，再做整图微调；
3. 在带标注验证集上输出分割指标；
4. 对当前未标注 holdout 图输出预测 mask 和几何量；
5. 为以后补测试标注、加 `test.py`、做论文实验预留结构。

---

## 2. 当前数据集真实情况

根据当前目录，数据定义如下。

### 2.1 有标注缺陷图

路径：

- `dataset_new/train/images/`
- `dataset_new/train/masks/`
- `dataset_new/train/images/*.json`

实际情况：

- 有 `81` 张缺陷图；
- 每张缺陷图都有同名 `.jpg`、`.png` mask、`.json` 标注；
- 图片尺寸统一为 `640x640`；
- mask 为单类二值缺陷分割；
- `json` 里标签目前可以统一按 `crack` 处理。

### 2.2 未标注 holdout 图

路径：

- `dataset_new/val/`

实际情况：

- 有 `42` 张 `.jpg` 图；
- 没有 mask；
- 当前只能做推理，不能做指标；
- 后续如果补标注，可以升级为正式测试集。

### 2.3 正常图

路径：

- `dataset_new/normal_crops_selected/`
- `dataset_new/normal_crops_selected/video_mapping.csv`

实际情况：

- 有 `5414` 张正常图；
- 都有 `video_id` 映射；
- 共来自 `41` 个视频；
- 可以按视频分组划分，不需要随机乱切。

### 2.4 备份目录

路径：

- `dataset_new/train/masks_backup_from_original_20260323_221111/`

实际情况：

- 这是旧 mask 备份；
- 整个项目都要显式排除；
- 任何扫描 `train` 目录的代码都不能误读它。

---

## 3. 当前版本的总体训练路线

当前版本固定采用两阶段训练。

### 阶段 1：Patch 训练

目标：

- 先让模型学会缺陷局部纹理、边界和背景差异；
- 避免因为整图前景太小，模型直接学成全背景。

训练输入：

- 从缺陷图中裁正样本 patch；
- 从缺陷图非缺陷区域裁困难负样本 patch；
- 从正常图裁普通负样本 patch。

### 阶段 2：整图微调

目标：

- 把阶段 1 学到的局部能力迁移到整张 `640x640` ROI；
- 输出完整二值 mask；
- 在验证集上做阈值和连通域搜索。

---

## 4. 项目目录结构

建议按下面结构手写：

```text
UNET_two_stage/
  dataset_new/
  manifests/
  src/
  scripts/
  configs/
  outputs/
```

具体说明如下。

### 4.1 `dataset_new/`

原始数据目录，只读，不修改。

### 4.2 `manifests/`

存放所有中间清单文件和 patch 索引文件。

### 4.3 `src/`

存放核心 Python 模块。

### 4.4 `scripts/`

存放各阶段的可执行脚本。

### 4.5 `configs/`

存放训练配置文件。

### 4.6 `outputs/`

存放训练输出、日志、模型和推理结果。

---

## 5. 第一步：建立目录骨架

先手动建立下面这些目录：

```text
manifests/
src/
scripts/
configs/
outputs/
```

这一步不要碰 `dataset_new/`。

---

## 6. 第二步：先写 `src/utils.py`

这个文件先写，因为后面很多脚本都会依赖它。

### 6.1 文件路径

```text
src/utils.py
```

### 6.2 需要实现的函数

#### `ensure_dir(path) -> None`

作用：

- 如果目录不存在则创建；
- 用于统一创建输出目录。

#### `set_seed(seed: int) -> None`

作用：

- 固定随机种子；
- 要同时设置：
  - `random`
  - `numpy`
  - `torch`
  - `torch.cuda`

#### `seed_worker(worker_id: int) -> None`

作用：

- 给 DataLoader worker 设随机种子；
- 保证多 worker 时增强和采样可复现。

#### `read_csv_rows(path) -> list[dict]`

作用：

- 统一读取 CSV，返回字典列表。

#### `write_csv_rows(path, rows, fieldnames) -> None`

作用：

- 统一写 CSV；
- 保证表头顺序固定。

#### `read_json(path) -> dict`

作用：

- 统一读取 JSON。

#### `save_json(path, obj) -> None`

作用：

- 统一写 JSON。

#### `load_yaml(path) -> dict`

作用：

- 统一读取配置文件。

---

## 7. 第三步：写 `scripts/prepare_dataset.py`

这是整个项目第一份核心脚本。

作用：

- 扫描原始数据；
- 关联视频映射；
- 生成统一 manifest；
- 按视频划分 defect train/val；
- 根据 defect 视频划分 normal train/val/future holdout。

### 7.1 文件路径

```text
scripts/prepare_dataset.py
```

### 7.2 需要实现的函数

#### `scan_labeled_defects(root) -> list[dict]`

逻辑：

1. 扫描 `dataset_new/train/images/*.jpg`
2. 为每张图寻找：
   - 同名 `json`
   - 同名 `masks/*.png`
3. 三者都存在才保留
4. 完全忽略 `masks_backup_from_original_20260323_221111`
5. 每条记录至少保存：
   - `sample_id`
   - `image_name`
   - `image_path`
   - `mask_path`
   - `json_path`
   - `sample_type='defect'`
   - `is_labeled=True`
   - `source_split='train'`

#### `scan_unlabeled_holdout(root) -> list[dict]`

逻辑：

1. 扫描 `dataset_new/val/*.jpg`
2. `mask_path=''`
3. `json_path=''`
4. `sample_type='defect_holdout_unlabeled'`
5. `is_labeled=False`
6. `source_split='val'`

#### `scan_normals(root) -> list[dict]`

逻辑：

1. 扫描 `dataset_new/normal_crops_selected/*.jpg`
2. 排除 `video_mapping.csv`
3. `mask_path=''`
4. `json_path=''`
5. `sample_type='normal'`
6. `is_labeled=False`
7. `source_split='normal_pool'`

#### `load_defect_train_mapping(path) -> dict[str, dict]`

作用：

- 读取 `图片来源视频映射/train_image_to_video.csv`
- 用 `image_name` 建索引

#### `load_defect_holdout_mapping(path) -> dict[str, dict]`

作用：

- 读取 `图片来源视频映射/val_image_to_video.csv`
- 用 `image_name` 建索引

#### `load_normal_mapping(path) -> dict[str, dict]`

作用：

- 读取 `normal_crops_selected/video_mapping.csv`
- 用 `image_name` 建索引

#### `attach_video_info(rows, mapping_dict) -> list[dict]`

逻辑：

1. 用 `image_name` 对接映射信息
2. 如果某图找不到映射，丢弃并统计
3. 每条记录补充：
   - `video_id`
   - `video_name`（如有）
   - `frame_id`（如有）

#### `build_master_manifest(defect_rows, holdout_rows, normal_rows) -> list[dict]`

作用：

- 合并三类数据，形成统一主表。

#### `split_defect_train_val_by_video(defect_rows, seed=42, val_ratio=0.2)`

逻辑：

1. 按 `video_id` 分组
2. 固定随机种子打乱组顺序
3. 逐组放入验证集，直到验证样本数接近总样本数的 `20%`
4. 剩余视频组进入训练集
5. 同一 `video_id` 不允许同时出现在 train 和 val

#### `split_normals_by_video(normal_rows, defect_train_videos, defect_val_videos, future_holdout_videos)`

逻辑：

1. `normal_val`：
   - `video_id` 属于 `defect_val_videos`
2. `normal_future_holdout`：
   - `video_id` 属于当前未标注 holdout 的视频组
3. `normal_train`：
   - 剩余全部正常图

这个规则的目的是：

- 验证时看同视频域下的正常误报；
- 未来 holdout 视频的正常图不提前进入训练；
- 保留训练正常图的多样性。

#### `write_all_manifests(...) -> None`

作用：

- 一次性把所有清单写到 `manifests/`

#### `main()`

主流程：

1. 扫描三类数据
2. 读取三份映射
3. 关联 `video_id`
4. 生成总 manifest
5. 按视频划分 defect train/val
6. 按视频划分 normal train/val/future holdout
7. 写 CSV 和 summary JSON

### 7.3 最终输出文件

这个脚本必须输出：

- `manifests/master_manifest.csv`
- `manifests/defect_labeled.csv`
- `manifests/defect_holdout_unlabeled.csv`
- `manifests/normal_pool.csv`
- `manifests/defect_train.csv`
- `manifests/defect_val.csv`
- `manifests/normal_train.csv`
- `manifests/normal_val.csv`
- `manifests/normal_future_holdout.csv`
- `manifests/split_summary.json`

### 7.4 `master_manifest.csv` 建议字段

```text
sample_id,image_name,image_path,mask_path,json_path,sample_type,is_labeled,video_id,source_split
```

### 7.5 这一阶段的验收标准

必须检查：

- `defect_labeled = 81`
- `defect_holdout_unlabeled = 42`
- `normal_pool = 5414`
- `defect_train` 和 `defect_val` 的 `video_id` 无交集
- `normal_future_holdout` 的 `video_id` 和当前 holdout 图一致
- 备份目录完全没有进入 manifest

---

## 8. 第四步：写 `src/datasets.py`

这个文件负责：

- 读图
- 读 mask
- 数据增强
- patch 裁切
- Dataset 类

### 8.1 文件路径

```text
src/datasets.py
```

### 8.2 需要实现的函数

#### `read_image_rgb(path) -> np.ndarray`

作用：

- 读 RGB 图像，返回 `H x W x 3`

#### `read_mask_binary(path, image_size=None) -> np.ndarray`

逻辑：

- 读灰度 mask
- 转成 `0/1`
- 如果路径为空，则报错或交给外部生成空 mask

#### `build_empty_mask(height, width) -> np.ndarray`

作用：

- 对正常图和未标注图返回全零 mask

#### `crop_patch(image, mask, x, y, crop_size, out_size)`

逻辑：

- 从 `(x, y)` 左上角裁一个正方形 patch
- 然后 resize 到 `out_size`

#### `build_stage1_train_transform(out_size)`

作用：

- 返回 patch 阶段训练增强

#### `build_stage1_eval_transform(out_size)`

作用：

- 返回 patch 阶段验证增强

#### `build_stage2_train_transform(image_size)`

作用：

- 返回整图训练增强

#### `build_stage2_eval_transform(image_size)`

作用：

- 返回整图验证增强

### 8.3 `ROIDataset`

这个类负责整图训练和验证。

#### 初始化参数建议

- `rows`
- `image_size`
- `transform`
- `return_empty_mask_for_unlabeled=True`

#### `__getitem__` 固定返回

```python
{
    "image": ...,
    "mask": ...,
    "sample_id": ...,
    "image_name": ...,
    "video_id": ...,
    "sample_type": ...,
    "is_labeled": ...,
}
```

逻辑约定：

- defect 样本读真实 mask
- normal 样本返回全零 mask
- holdout 样本也返回全零 mask，但只用于推理，不参与指标

### 8.4 `PatchDataset`

这个类负责阶段 1 训练。

#### 初始化参数建议

- `patch_rows`
- `transform`

#### `__getitem__` 固定返回

```python
{
    "image": ...,
    "mask": ...,
    "patch_id": ...,
    "base_sample_id": ...,
    "patch_type": ...,
    "video_id": ...,
}
```

---

## 9. 第五步：写 `scripts/build_patch_index.py`

这个脚本只负责生成 patch 索引。

### 9.1 文件路径

```text
scripts/build_patch_index.py
```

### 9.2 需要实现的函数

#### `mask_to_bbox(mask)`

作用：

- 根据前景 mask 计算最小外接框

#### `clip_crop_window(center_x, center_y, crop_size, image_w, image_h)`

作用：

- 把裁块窗口限制在图像范围内

#### `make_positive_center_patches(row, mask)`

规则：

- 以缺陷中心为中心裁块
- 每张图生成 `2` 个 patch
- `crop_size` 从 `[320, 384, 448]` 中选

#### `make_positive_shift_patches(row, mask)`

规则：

- 在缺陷中心附近平移
- 每张图生成 `2` 个 patch
- 平移范围建议 `±24` 像素
- 缺陷必须完整保留

#### `make_positive_context_patches(row, mask)`

规则：

- 缺陷不要求在正中心
- 但必须完整保留
- 每张图生成 `2` 个 patch

#### `make_hard_negative_patches(row, mask)`

规则：

- 从缺陷图非缺陷区域裁块
- patch 内不能有任何正像素
- 与缺陷 bbox 保持一定安全边
- 每张图生成 `4` 个 patch

#### `make_normal_negative_patches(normal_rows, target_count, seed)`

规则：

- 从 `normal_train` 或 `normal_val` 中采样
- patch 大小与正样本一致
- 总数量与困难负样本数接近

#### `build_train_patch_index(...)`

作用：

- 生成训练 patch 索引

#### `build_val_patch_index(...)`

作用：

- 生成验证 patch 索引

#### `main()`

主流程：

1. 读取 defect/normal manifest
2. 为训练集生成 patch 索引
3. 为验证集生成 patch 索引
4. 写 CSV

### 9.3 输出文件

- `manifests/stage1_patch_train_index.csv`
- `manifests/stage1_patch_val_index.csv`

### 9.4 patch index 建议字段

```text
patch_id,base_sample_id,image_path,mask_path,patch_type,video_id,crop_x,crop_y,crop_size,out_size
```

### 9.5 这一阶段的验收标准

必须做到：

- 正样本 patch 内有缺陷
- 困难负样本 patch 内没有任何缺陷像素
- `stage1_patch_val_index.csv` 只来源于验证清单
- 正负样本比例合理，不让 normal 负样本压倒一切

---

## 10. 第六步：写 `src/model.py`

### 10.1 文件路径

```text
src/model.py
```

### 10.2 需要实现的类

#### `ConvBlock`

作用：

- 经典 `conv + bn + relu` 组合

#### `DecoderBlock`

作用：

- 上采样
- 拼接 skip connection
- 卷积融合

#### `UNetResNet34`

作用：

- 编码器用 `torchvision.models.resnet34`
- 解码器自己写
- 输出单通道 logits

#### `build_model(pretrained=True)`

作用：

- 构建模型实例

### 10.3 形状要求

输入：

- `B x 3 x H x W`

输出：

- `B x 1 x H x W`

### 10.4 这一阶段自检

至少做：

- 输入 `2 x 3 x 384 x 384`，输出尺寸正确
- 输入 `2 x 3 x 640 x 640`，输出尺寸正确

---

## 11. 第七步：写 `src/losses.py`

### 11.1 文件路径

```text
src/losses.py
```

### 11.2 需要实现的类

#### `DiceLoss`

逻辑：

- 输入 logits 和 target
- 内部先做 `sigmoid`
- 用 `eps` 避免除零

#### `BCEDiceLoss`

逻辑：

- `BCEWithLogitsLoss + DiceLoss`
- 可配置：
  - `bce_weight`
  - `dice_weight`
  - `pos_weight`

### 11.3 默认组合

```text
总损失 = 0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss
```

### 11.4 验收标准

必须保证：

- 全背景 batch 不出 `NaN`
- 前向和反向都正常
- loss 会随预测变好而下降

---

## 12. 第八步：写 `src/postprocess.py`

### 12.1 文件路径

```text
src/postprocess.py
```

### 12.2 需要实现的函数

#### `logits_to_probs(logits)`

作用：

- `sigmoid`

#### `probs_to_binary_mask(probs, threshold)`

作用：

- 按阈值二值化

#### `remove_small_components(mask, min_area)`

作用：

- 去掉小连通域

#### `apply_postprocess(probs, threshold, min_area)`

作用：

- 二值化 + 连通域过滤的一体化接口

### 12.3 实现要求

建议用：

- `cv2.connectedComponentsWithStats`

因为这个接口可以直接拿到面积。

---

## 13. 第九步：写 `src/geometry.py`

### 13.1 文件路径

```text
src/geometry.py
```

### 13.2 需要实现的函数

#### `mask_to_connected_regions(mask)`

作用：

- 分离各个连通域

#### `region_to_bbox(region_mask)`

作用：

- 算单个连通域外接框

#### `compute_region_geometry(region_mask)`

作用：

- 算：
  - 面积
  - 周长
  - bbox 宽高
  - 长宽比

#### `export_regions_for_image(image_name, mask)`

作用：

- 输出某张图的所有连通域几何量

### 13.3 几何表字段建议

- `image_name`
- `region_id`
- `x_min`
- `y_min`
- `x_max`
- `y_max`
- `area_px`
- `perimeter_px`
- `bbox_w`
- `bbox_h`
- `aspect_ratio`

---

## 14. 第十步：写 `src/metrics.py`

### 14.1 文件路径

```text
src/metrics.py
```

### 14.2 需要实现的函数

#### `dice_score(pred, target)`
#### `iou_score(pred, target)`
#### `compute_defect_seg_metrics(pred_masks, gt_masks)`
#### `compute_normal_fpr(pred_masks)`
#### `compute_defect_image_recall(pred_masks)`
#### `summarize_metrics(per_image_rows)`
#### `grid_search_postprocess(prob_maps, gt_masks, threshold_candidates, min_area_candidates, sample_types)`

### 14.3 指标定义固定

#### `defect_dice`

只在 GT 有缺陷的图上算平均 Dice。

#### `defect_iou`

只在 GT 有缺陷的图上算平均 IoU。

#### `defect_image_recall`

GT 是缺陷图时，只要最终预测有任意正像素，就算召回。

#### `normal_fpr`

正常图只要最终预测有任意正像素，就算误报。

### 14.4 阈值搜索规则固定

1. 先筛掉 `normal_fpr > 0.10` 的组合；
2. 在剩余组合里选择 `defect_dice` 最大的；
3. 如果没有任何组合满足限制，就选 `normal_fpr` 最低的组合。

---

## 15. 第十一步：写 `src/trainer.py`

### 15.1 文件路径

```text
src/trainer.py
```

### 15.2 需要实现的内容

#### `build_optimizer(model, cfg)`

逻辑：

- 编码器和解码器分组学习率
- 编码器：
  - `encoder_lr`
- 解码器和输出头：
  - `decoder_lr`

#### `build_scheduler(optimizer, cfg)`

建议：

- `ReduceLROnPlateau`

#### `save_checkpoint(path, state)`
#### `load_checkpoint(path, model, optimizer=None, scheduler=None)`

#### `train_one_epoch(model, loader, criterion, optimizer, scaler, device)`

返回：

- `loss`
- `lr_encoder`
- `lr_decoder`

#### `predict_on_loader(model, loader, device)`

作用：

- 验证和推理时统一调用

#### `validate_stage1(model, loader, criterion, device)`

输出：

- `val_loss`
- `patch_dice`

#### `validate_stage2(model, loader, device, threshold_candidates, min_area_candidates)`

输出：

- `defect_dice`
- `defect_iou`
- `defect_image_recall`
- `normal_fpr`
- `best_threshold`
- `best_min_area`

#### `EarlyStopper`

逻辑：

- 监控 `defect_dice`

---

## 16. 第十二步：写 `configs/stage1.yaml`

### 16.1 文件路径

```text
configs/stage1.yaml
```

### 16.2 建议字段

```yaml
seed: 42
patch_out_size: 384
batch_size: 8
epochs: 30
encoder_lr: 5.0e-5
decoder_lr: 2.0e-4
weight_decay: 1.0e-4
bce_weight: 0.5
dice_weight: 0.5
pos_weight: 12.0
amp: true
freeze_encoder_epochs: 3
early_stop_patience: 12
num_workers: 4
train_index_path: manifests/stage1_patch_train_index.csv
val_index_path: manifests/stage1_patch_val_index.csv
save_dir: outputs/stage1
```

---

## 17. 第十三步：写 `configs/stage2.yaml`

### 17.1 文件路径

```text
configs/stage2.yaml
```

### 17.2 建议字段

```yaml
seed: 42
image_size: 640
batch_size: 4
epochs: 25
encoder_lr: 2.0e-5
decoder_lr: 1.0e-4
weight_decay: 1.0e-4
bce_weight: 0.5
dice_weight: 0.5
pos_weight: 12.0
amp: true
early_stop_patience: 12
num_workers: 4
threshold_candidates: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
min_area_candidates: [5, 10, 20]
defect_train_manifest: manifests/defect_train.csv
defect_val_manifest: manifests/defect_val.csv
normal_train_manifest: manifests/normal_train.csv
normal_val_manifest: manifests/normal_val.csv
stage1_checkpoint: outputs/stage1/best_stage1.pt
save_dir: outputs/stage2
```

---

## 18. 第十四步：写 `scripts/train_stage1.py`

### 18.1 文件路径

```text
scripts/train_stage1.py
```

### 18.2 需要实现的函数

#### `parse_args()`

作用：

- 读 `--config`

#### `build_stage1_loaders(cfg)`

作用：

- 构建训练和验证 DataLoader

#### `main()`

流程：

1. 读 config
2. 设 seed
3. 建 patch dataset 和 loader
4. 建模型
5. 建 loss
6. 建 optimizer 和 scheduler
7. 建 AMP scaler
8. 前 `3` 个 epoch 冻结 encoder
9. 每轮训练后验证
10. 保存：
    - `best_stage1.pt`
    - `last_stage1.pt`
    - `history.csv`

### 18.3 best 模型标准

- 以 `patch_dice` 最大为 best

---

## 19. 第十五步：写 `scripts/train_stage2.py`

### 19.1 文件路径

```text
scripts/train_stage2.py
```

### 19.2 需要实现的函数

#### `parse_args()`

#### `sample_normal_rows(normal_rows, k, seed)`

作用：

- 每个 epoch 从 `normal_train` 里采样一部分正常图

#### `build_epoch_train_rows(defect_train_rows, normal_train_rows, epoch_seed)`

逻辑：

- 每个 epoch 使用全部 `defect_train`
- 再从 `normal_train` 随机采样与 defect 数量接近的正常图
- 拼成该 epoch 的整图训练集

#### `build_stage2_train_loader(rows, cfg)`
#### `build_stage2_val_loader(defect_val_rows, normal_val_rows, cfg)`

#### `main()`

流程：

1. 读 config
2. 加载 stage1 最优权重
3. 每个 epoch 重采样一份 `normal_train`
4. 构建本轮整图训练集
5. 验证集固定为：
   - `defect_val`
   - `normal_val`
6. 每轮验证时做阈值和最小连通域搜索
7. 保存：
   - `best_stage2.pt`
   - `last_stage2.pt`
   - `history.csv`

### 19.3 best 模型规则

固定为：

1. 优先满足 `normal_fpr <= 0.10`
2. 在满足条件的 checkpoint 中选择 `defect_dice` 最大的

---

## 20. 第十六步：写 `scripts/evaluate_val.py`

### 20.1 文件路径

```text
scripts/evaluate_val.py
```

### 20.2 作用

- 对正式验证集输出最终验证报告

### 20.3 逻辑

1. 读取 `best_stage2.pt`
2. 读取 `defect_val` 和 `normal_val`
3. 推理得到概率图
4. 用配置中的候选阈值和最小面积搜索最佳后处理参数
5. 输出验证报告

### 20.4 输出文件

- `outputs/stage2/val_metrics.json`
- `outputs/stage2/val_per_image.csv`

---

## 21. 第十七步：写 `scripts/infer_holdout.py`

### 21.1 文件路径

```text
scripts/infer_holdout.py
```

### 21.2 作用

- 对当前未标注 holdout 图做最终推理

### 21.3 需要实现的函数

#### `parse_args()`
#### `save_prob_map(path, probs)`
#### `save_binary_mask(path, mask)`
#### `main()`

### 21.4 `main()` 逻辑

1. 读取 `defect_holdout_unlabeled.csv`
2. 读取 `best_stage2.pt`
3. 读取 `val_metrics.json` 里的最佳：
   - `threshold`
   - `min_area`
4. 推理未标注图
5. 输出：
   - 概率图
   - 二值 mask
   - 几何量表

### 21.5 输出目录

```text
outputs/stage2/holdout/
  prob_maps/
  masks/
  geometry.csv
```

---

## 22. 推荐手写顺序

按下面顺序写，最稳：

1. 建目录
2. 写 `src/utils.py`
3. 写 `scripts/prepare_dataset.py`
4. 跑出全部 manifest
5. 写 `src/datasets.py`
6. 写 `scripts/build_patch_index.py`
7. 跑出 patch index
8. 写 `src/model.py`
9. 写 `src/losses.py`
10. 写 `src/postprocess.py`
11. 写 `src/geometry.py`
12. 写 `src/metrics.py`
13. 写 `src/trainer.py`
14. 写两个 config
15. 写 `scripts/train_stage1.py`
16. 写 `scripts/train_stage2.py`
17. 写 `scripts/evaluate_val.py`
18. 写 `scripts/infer_holdout.py`

---

## 23. 每一步的最小自检

### 23.1 `prepare_dataset.py`

检查：

- 各 manifest 数量是否正确
- `video_id` 是否无泄漏
- 备份目录是否被排除

### 23.2 `datasets.py`

检查：

- defect 样本 `mask.sum() > 0`
- normal 样本 `mask.sum() == 0`
- holdout 样本 `mask.sum() == 0`

### 23.3 `build_patch_index.py`

检查：

- 随机画几张正样本 patch
- 随机画几张 hard negative patch
- 保证 patch 类型没有错

### 23.4 `model.py`

检查：

- `384` 和 `640` 输入都能正常前向

### 23.5 `losses.py`

检查：

- 全背景 batch 不出 NaN

### 23.6 `trainer.py`

检查：

- 先跑 1 个 epoch smoke test
- 再跑正式训练

### 23.7 `evaluate_val.py`

检查：

- `best_threshold`
- `best_min_area`
- `defect_dice`
- `normal_fpr`

### 23.8 `infer_holdout.py`

检查：

- `42` 张未标注图都有输出文件

---

## 24. 当前版本的边界

当前版本明确不做：

1. 不做 `test.py`
2. 不做 5 折交叉验证
3. 不做多类别分割
4. 不做复杂骨干替换
5. 不做物理尺寸标定
6. 不做扩散模型增强
7. 不直接改原始数据目录

---

## 25. 以后怎么升级

如果未来 `dataset_new/val` 补了标注，升级方式是：

1. 在 manifest 阶段新增 `defect_test.csv`
2. 在 normal 里启用 `normal_future_holdout.csv`
3. 新增 `scripts/test.py`
4. 其余代码结构基本不用推翻

也就是说，当前这个版本是一个可直接过渡到正式 test 流程的最小可运行工程。

---
