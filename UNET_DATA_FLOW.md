# UNET Two-Stage Data Flow

本文按数据流说明 Alvis 服务器上当前 `UNET_two_stage` 的训练、验证、后处理和推理流程。

服务器项目目录：

```text
/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/UNET_two_stage
```

服务器数据目录：

```text
/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/dataset0505_crop640_roi
```

当前服务器 manifest 摘要：

```text
total:    444
trainval: 270
holdout:  174

trainval:
  camera crack: 126
  camera normal: 30
  phone  crack: 85
  phone  normal: 29

holdout:
  camera broken: 44
  camera crack: 31
  camera normal: 8
  phone  broken: 63
  phone  crack: 21
  phone  normal: 7

fold_counts:
  fold0: 70
  fold1: 68
  fold2: 66
  fold3: 66
```

当前已跑/准备的主要 Alvis 配置：

```text
Stage1 P0 A40:
  configs/stage1_p0_a40.yaml
  outputs/experiments/p0_a40_20260508/stage1/fold{fold}

Stage2 P0 A40 baseline:
  configs/stage2_p0_a40.yaml
  outputs/experiments/p0_a40_20260508/stage2/fold{fold}

Stage2 P0 A40 e50 bs48 rerun:
  configs/stage2_p0_a40_e50_bs48.yaml
  outputs/experiments/p0_a40_e50_bs48_20260509/stage2/fold{fold}
```

## 总览

```text
原始 ROI 数据
  -> prepare_samples.py
  -> samples.csv + generated_masks/
  -> trainval 4-fold + holdout
  -> build_patch_index.py
  -> Stage1 patch indexes
  -> train_stage1.py
  -> best_stage1.pt
  -> train_stage2.py
  -> best_stage2.pt + per-fold validation metrics
  -> search_oof_postprocess.py
  -> global threshold/min_area
  -> infer_holdout.py
  -> holdout probability maps and masks
```

## 1. 原始数据

默认数据目录是项目同级的 `dataset0505_crop640_roi/`：

```text
dataset0505_crop640_roi/
  device_x/
    crack/
      image.png
      image.json
    normal/
      image.png
    broken/
      image.png
```

三类数据的角色：

- `crack`: 有 LabelMe JSON，作为有标注缺陷样本。
- `normal`: 没有缺陷，作为负样本，训练时使用全 0 mask。
- `broken`: 当前没有可靠 mask，默认只进入 holdout，用于推理和定性检查，不进入 Stage1/Stage2 训练。

## 2. Manifest 和 Mask 生成

入口：

```bash
python scripts/prepare_samples.py
```

主要逻辑：

1. 扫描原始 ROI 数据目录。
2. 对 `crack` 的 LabelMe polygon 生成二值 mask，保存到 `generated_masks/`。
3. 对 `normal` 保留空 `mask_path`，后续 Dataset 会生成全 0 mask。
4. 对 `broken` 标记为 `broken_unlabeled`，直接放入 `holdout`。
5. 默认将 20% 的 `crack + normal` 也保留为 inference-only `holdout`。
6. 剩余 `crack + normal` 组成 `trainval`，再分成 `cv_fold=0,1,2,3`。

产物：

```text
manifests/samples.csv
manifests/samples_summary.json
generated_masks/
```

`samples.csv` 是后续全流程的主索引，关键字段包括：

- `sample_id`
- `image_path`
- `mask_path`
- `sample_type`
- `defect_class`
- `split`
- `cv_fold`
- `holdout_reason`

数据使用约束：

```text
split == trainval -> Stage1/Stage2 训练与交叉验证
split == holdout  -> 训练不使用，只做最终推理或测试
```

## 3. Trainval / Holdout 划分

默认划分来自：

```bash
python scripts/prepare_samples.py --test-ratio 0.20 --test-seed 2026
```

概念上分成两层：

```text
trainval:
  用于 4-fold cross-validation。

holdout:
  独立保留，训练和调参阶段不应使用。
```

每个 fold 的训练/验证规则：

```text
val   = cv_fold == 当前 fold
train = split == trainval 且 cv_fold != 当前 fold
```

代码入口是 `src/samples.py` 的 `split_samples_for_fold()`。

## 4. Stage1 Patch Index

入口：

```bash
python scripts/build_patch_index.py --config configs/stage1.yaml
```

Stage1 不直接训练整图，而是先把 trainval 图像转换成 patch index。每个 patch row 描述：

```text
image_path + mask_path + crop_x + crop_y + crop_size + out_size + patch_type
```

对 `crack` 图像：

1. 读取二值 mask。
2. 找 connected components。
3. 基于缺陷区域生成正样本 patch。
4. 基于缺陷附近和远离缺陷区域生成负样本 patch。

主要 patch 类型：

```text
positive_boundary
positive_shift
positive_center
positive_context
near_miss_negative
hard_negative
normal_negative
```

含义：

- `positive_*`: patch 内包含缺陷。
- `near_miss_negative`: 靠近缺陷但不含缺陷，用于学习边界附近的假阳性。
- `hard_negative`: 远离缺陷的缺陷图负样本。
- `normal_negative`: normal 图像随机裁剪出的负样本。

产物：

```text
manifests/stage1_fold0_train_index.csv
manifests/stage1_fold0_val_index.csv
...
manifests/stage1_patch_summary.json
```

## 5. Stage1 Patch-Level Training

入口：

```bash
python scripts/train_stage1.py --config configs/stage1.yaml --fold 0
```

数据流：

```text
stage1_fold*_train_index.csv
  -> PatchDataset
  -> crop original image and mask
  -> resize to patch_out_size
  -> augmentation
  -> ResNet34-UNet
  -> patch logits
  -> BCE + Dice loss
```

模型：

```text
ResNet34 encoder + UNet decoder
```

定义在：

```text
src/model.py
```

损失：

```text
BCEDiceLoss = bce_weight * BCEWithLogits + dice_weight * DiceLoss
```

定义在：

```text
src/losses.py
```

Stage1 的目标是先学习局部 patch 里的缺陷外观，并减少局部负样本误检。验证时不只看 Dice，还会看负 patch 的 false positive rate。

Stage1 score：

```text
stage1_score =
  patch_dice_pos_only
  - stage1_negative_fpr_penalty * max(0, negative_patch_fpr - stage1_target_negative_fpr)
```

当前 P0 A40 配置示例：

```text
configs/stage1_p0_a40.yaml
batch_size: 128
epochs: 30
patch_out_size: 384
stage1_sampler_mode: balanced
stage1_positive_ratio: 0.50
stage1_defect_negative_ratio: 0.25
stage1_normal_ratio: 0.25
stage1_use_replay: true
```

产物：

```text
outputs/.../stage1/fold*/best_stage1.pt
outputs/.../stage1/fold*/last_stage1.pt
outputs/.../stage1/fold*/history.csv
```

## 6. Stage1 Replay

Stage1 训练过程中会周期性扫描已有 patch，找困难样本并回放。

数据流：

```text
base patch index
  -> current Stage1 model prediction
  -> find false negatives / false positives
  -> replay rows
  -> append to later epoch training rows
```

Replay 类型：

```text
replay_positive_fn
replay_positive_hard
replay_defect_negative_fp
replay_normal_negative_fp
```

产物：

```text
outputs/.../stage1/fold*/replay/replay_latest.csv
outputs/.../stage1/fold*/replay/replay_epochXXX.csv
outputs/.../stage1/fold*/replay/replay_latest_summary.json
```

## 7. Stage2 Full-Image Training

入口：

```bash
python scripts/train_stage2.py --config configs/stage2.yaml --fold 0
```

Stage2 的输入单位是整张 ROI 图，而不是 patch。

数据流：

```text
samples.csv
  -> split_samples_for_fold()
  -> defect_train_rows / defect_val_rows
  -> normal_train_rows / normal_val_rows
  -> ROIDataset
  -> full 640x640 image + mask
```

初始化：

```text
load outputs/.../stage1/fold{fold}/best_stage1.pt
```

也就是说 Stage2 接着 Stage1 的权重继续训练同一个 ResNet34-UNet，但训练尺度从 patch 变成整图。

每个 epoch 的训练行：

```text
all defect_train_rows
+ sampled normal_train_rows
+ sampled hard_normal_rows
```

normal 预算：

```text
normal_budget_count = defect_train_count * random_normal_k_factor
```

当前 P0 A40 Stage2 配置示例：

```text
configs/stage2_p0_a40_e50_bs48.yaml
image_size: 640
batch_size: 48
epochs: 50
random_normal_k_factor: 1.0
use_hard_normal_replay: true
stage2_hard_normal_ratio: 0.40
```

产物：

```text
outputs/.../stage2/fold*/best_stage2.pt
outputs/.../stage2/fold*/last_stage2.pt
outputs/.../stage2/fold*/history.csv
```

## 8. Stage2 Hard Normal Replay

Stage2 的 hard normal replay 用来压低 normal 图上的误检。

数据流：

```text
normal_train_rows
  -> current Stage2 model
  -> probability map
  -> threshold + min_area
  -> false positive pixels/components
  -> hard_score
  -> hard normal pool
  -> later epochs replace part of random normal samples
```

hard normal 评分主要来自：

```text
hard_score =
  largest_fp_area * 1000000
  + fp_pixel_count
  + max_prob
```

当前配置：

```text
hard_normal_warmup_epochs: 2
hard_normal_refresh_every: 2
stage2_hard_normal_ratio: 0.40
hard_normal_pool_factor: 3.0
```

含义：

- 前 2 个 epoch 不做 hard normal。
- 之后每隔 2 个 epoch 扫描 normal 训练图。
- 每个 epoch 的 normal 预算中约 40% 来自 hard normal pool。

产物：

```text
outputs/.../stage2/fold*/hard_normal/hard_normal_latest.csv
outputs/.../stage2/fold*/hard_normal/hard_normal_epochXXX.csv
outputs/.../stage2/fold*/hard_normal/hard_normal_latest_summary.json
```

## 9. Stage2 Validation 和模型选择

Stage2 每个 epoch 都会在当前 fold 的 validation set 上评估。

验证数据：

```text
defect_val_rows + normal_val_rows
```

验证过程：

```text
model logits
  -> sigmoid probability map
  -> threshold grid
  -> min_area grid
  -> binary mask
  -> metrics
```

主要指标：

```text
defect_dice
defect_iou
defect_image_recall
normal_fpr
normal_fp_count
normal_fp_pixel_sum
normal_largest_fp_area_*
```

后处理参数搜索：

```text
threshold_grid_start: 0.10
threshold_grid_end: 0.90
threshold_grid_step: 0.02
min_area_grid: [0, 8, 16, 24, 32, 48]
```

Stage2 score：

```text
stage2_score =
  defect_dice
  - lambda_fpr_penalty * max(0, normal_fpr - target_normal_fpr)
```

模型选择逻辑：

1. 优先选择 `normal_fpr <= target_normal_fpr` 的模型。
2. 在满足 normal FPR 的模型中，选择 defect Dice 更高的。
3. 如果都不满足 normal FPR，则比较 `stage2_score`。

默认目标：

```text
target_normal_fpr: 0.10
lambda_fpr_penalty: 2.0
```

## 10. Per-Fold Validation Export

入口：

```bash
python scripts/evaluate_val.py --config configs/stage2.yaml --fold 0
```

`train_stage2.py` 默认会在训练结束后自动运行它：

```text
auto_evaluate_after_train: true
```

数据流：

```text
best_stage2.pt
  -> validation prediction
  -> raw threshold=0.5 evaluation
  -> threshold/min_area grid search
  -> val metrics and per-image CSV
```

产物：

```text
outputs/.../stage2/fold*/val_metrics.json
outputs/.../stage2/fold*/val_per_image.csv
outputs/.../stage2/fold*/val_postprocess_search.csv
```

## 11. Pooled OOF Post-Process Search

入口：

```bash
python scripts/search_oof_postprocess.py --config configs/stage2.yaml --folds 0,1,2,3
```

OOF 表示 out-of-fold。每张 trainval 图只用“没有训练过它的 fold 模型”预测一次。

数据流：

```text
fold0 best_stage2.pt -> predict fold0 val
fold1 best_stage2.pt -> predict fold1 val
fold2 best_stage2.pt -> predict fold2 val
fold3 best_stage2.pt -> predict fold3 val
  -> pool all validation predictions
  -> search one global threshold/min_area
```

产物：

```text
outputs/.../stage2/oof_global_postprocess.json
outputs/.../stage2/oof_global_postprocess_search.csv
outputs/.../stage2/oof_per_image.csv
```

后续 holdout inference 会优先使用这个全局后处理参数。

## 12. Holdout Inference

入口：

```bash
python scripts/infer_holdout.py --config configs/stage2.yaml --fold 0
```

输入：

```text
samples.csv 中 split == holdout 的样本
```

后处理参数来源优先级：

```text
1. global_postprocess_path 指向的 oof_global_postprocess.json
2. 当前 fold 的 val_metrics.json
```

数据流：

```text
holdout image
  -> best_stage2.pt
  -> probability map
  -> raw binary mask
  -> min_area filtered mask
  -> inference_summary.csv
```

产物：

```text
outputs/.../stage2/fold*/holdout/prob_maps/
outputs/.../stage2/fold*/holdout/raw_binary_masks/
outputs/.../stage2/fold*/holdout/masks/
outputs/.../stage2/fold*/holdout/inference_summary.csv
```

注意：

- `holdout` 中的 `crack + normal` 可以作为最终测试集使用。
- `broken` 当前没有可靠 GT mask，因此主要用于预测结果检查和定性分析。
- 当前 `infer_holdout.py` 主要负责导出预测，不是完整的 holdout metric 评估脚本。

## 13. 多 Fold / 多 GPU 执行方式

通用入口：

```bash
bash scripts/train_all_folds.sh --gpus 0,1,2,3
```

P0 A40 实验入口示例：

```bash
bash scripts/run_p0_a40_parallel.sh
```

P0 A40 的并行方式：

```text
fold0 -> GPU0
fold1 -> GPU1
fold2 -> GPU2
fold3 -> GPU3
```

每个 fold 内部：

```text
Stage1 fold
  -> Stage2 fold
```

所有 fold 结束后：

```text
search_oof_postprocess.py
```

这里不是 DDP。UNET 当前多卡利用方式是“一个 fold 一张 GPU”，四个 fold 并行训练。

## 14. 关键文件索引

数据准备：

```text
scripts/prepare_samples.py
scripts/build_patch_index.py
src/samples.py
src/datasets.py
```

训练：

```text
scripts/train_stage1.py
scripts/train_stage2.py
src/model.py
src/losses.py
src/trainer.py
src/mining.py
```

验证和后处理：

```text
scripts/evaluate_val.py
scripts/search_oof_postprocess.py
src/metrics.py
```

推理：

```text
scripts/infer_holdout.py
```

配置：

```text
configs/stage1.yaml
configs/stage2.yaml
configs/stage1_p0_a40.yaml
configs/stage2_p0_a40.yaml
configs/stage2_p0_a40_e50_bs48.yaml
```

## 15. 当前流程的核心约束

1. `holdout` 不应该进入 Stage1/Stage2 训练和 CV validation。
2. Stage1 使用 patch-level 数据，Stage2 使用 full-image 数据。
3. Stage2 依赖同 fold 的 `best_stage1.pt` 初始化。
4. normal 图在 Stage2 中不是一次性全量加入，而是按预算采样。
5. hard normal replay 只从 train fold 的 normal 图中挖掘。
6. threshold/min_area 应优先使用 pooled OOF 的全局设置，而不是单 fold 的局部最优。
7. UNET 多 GPU 方式是 fold-level parallelism，不是 DDP。
