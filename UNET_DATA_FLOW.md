# U-Net Two-Stage Data Flow and Mechanism Summary

本文档按数据流顺序完整说明 `UNET_two_stage` 项目的数据、模型、训练、验证、后处理、推理和实验机制。写作目标是给论文方法章节、实验章节和消融实验章节提供可直接改写的中英双语材料。

This document describes the complete data flow and mechanism design of the `UNET_two_stage` project, including data preparation, model architecture, training, validation, post-processing, inference, and experiments. It is written as bilingual material that can be adapted for the method and experiment sections of a thesis.

## 0. Project Snapshot / 项目快照

中文：

当前项目实现的是一个两阶段 U-Net 缺陷分割流程。Stage1 在局部 patch 上学习缺陷外观和困难负样本；Stage2 加载同一 fold 的 Stage1 最优权重，在完整 ROI 图像上继续训练。最终通过 pooled out-of-fold validation 搜索一个全局 `threshold/min_area` 后处理参数，用于验证总结和 holdout 推理。

English:

The project implements a two-stage U-Net defect segmentation pipeline. Stage1 learns local defect appearance and difficult negatives from cropped patches. Stage2 loads the best Stage1 checkpoint from the same fold and continues training on full ROI images. Final post-processing parameters are selected by pooling out-of-fold validation predictions and searching for one global `threshold/min_area` pair.

服务器路径 / Server paths:

```text
U-Net project:
/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/UNET_two_stage

ROI dataset:
/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/dataset0505_crop640_roi
```

本地项目路径 / Local project path:

```text
/Users/yangyi/Desktop/masterthesis/UNET_two_stage
```

核心结论 / Main conclusion:

```text
Best current U-Net setting:
configs/stage2_p0_a40_e50_bs48_pw12.yaml

Pooled OOF metrics:
Dice                 0.756876
IoU                  0.633878
defect image recall  0.995261
normal FPR           0.000000
threshold            0.80
min_area             24
```

中文：

当前最稳的 U-Net 结果是 Stage2 `pos_weight=12`，在 pooled OOF 上达到最高 Dice，同时保持 defect image recall 接近 1，并且 normal false positive rate 为 0。后续 deep supervision、boundary auxiliary loss 和 normal false-positive loss 都没有稳定超过该 baseline。

English:

The strongest current U-Net result is the Stage2 `pos_weight=12` baseline. It gives the best balanced pooled OOF result, with the highest Dice among the tested stable settings, near-perfect defect-image recall, and zero normal false-positive rate. Deep supervision, boundary auxiliary loss, and normal false-positive loss did not reliably improve over this baseline.

## 1. End-to-End Flow / 总体数据流

中文：

整个系统从原始 ROI 图像开始，经过样本索引、mask 生成、fold 划分、Stage1 patch 训练、Stage2 整图训练、OOF 后处理搜索，最后导出 holdout 预测。可以按下面的数据流理解：

English:

The full system starts from raw ROI images, builds manifests and masks, assigns cross-validation folds, trains a patch-level Stage1 model, fine-tunes a full-image Stage2 model, searches pooled OOF post-processing parameters, and finally exports holdout predictions. The data flow is:

```text
Raw ROI dataset
  -> scripts/prepare_samples.py
  -> manifests/samples.csv + generated_masks/
  -> trainval 4-fold split + holdout split
  -> scripts/build_patch_index.py
  -> Stage1 patch indexes
  -> scripts/train_stage1.py
  -> best_stage1.pt
  -> scripts/train_stage2.py
  -> best_stage2.pt + per-fold validation exports
  -> scripts/search_oof_postprocess.py
  -> one global threshold/min_area pair
  -> scripts/infer_holdout.py
  -> holdout probability maps and binary masks
```

论文写法 / Thesis wording:

中文：

本文采用两阶段训练策略。第一阶段在局部 patch 上训练 U-Net，使模型能够在缺陷像素稀疏的条件下充分观察裂纹区域及其邻近背景；第二阶段将第一阶段权重迁移到完整 ROI 图像上进行微调，使模型恢复整图上下文并学习 normal 图像上的误检抑制。最终采用 pooled out-of-fold validation 选择统一的后处理参数，从而避免每个 fold 使用局部最优阈值带来的评估偏差。

English:

A two-stage training strategy is used. In the first stage, the U-Net is trained on local patches so that the model can observe sparse crack pixels and nearby background regions more frequently. In the second stage, the Stage1 weights are transferred to full ROI images, allowing the model to recover global image context and suppress false positives on normal samples. A single global post-processing configuration is selected on pooled out-of-fold validation predictions to avoid fold-specific threshold tuning bias.

## 2. Raw Dataset / 原始数据集

中文：

原始 ROI 数据默认位于项目同级目录 `dataset0505_crop640_roi/`。当前数据来自两个设备来源：`camera` 和 `phone`。每个设备下包含三个目录：`crack`、`normal`、`broken`。

English:

The raw ROI dataset is expected under the sibling directory `dataset0505_crop640_roi/`. The current dataset contains two sources, `camera` and `phone`. Each source contains three folders: `crack`, `normal`, and `broken`.

目录结构 / Directory layout:

```text
dataset0505_crop640_roi/
|- camera/
|  |- crack/
|  |- normal/
|  `- broken/
`- phone/
   |- crack/
   |- normal/
   `- broken/
```

类别角色 / Class roles:

| folder | 中文说明 | English description |
| --- | --- | --- |
| `crack` | 有 LabelMe JSON 标注，是监督训练中的缺陷样本。 | Labeled defect images with same-stem LabelMe JSON files. |
| `normal` | 无缺陷图像，训练和验证时使用全 0 mask。 | Negative images. They are paired with all-zero masks during training and validation. |
| `broken` | 当前没有可靠 pixel mask，只进入 holdout，不参与监督训练或 CV 验证。 | Unlabeled holdout images. They are not used for supervised training or cross-validation. |

当前数据量 / Current counts:

| source | crack | normal | broken | total |
| --- | ---: | ---: | ---: | ---: |
| camera | 157 | 38 | 44 | 239 |
| phone | 106 | 36 | 63 | 205 |
| total | 263 | 74 | 107 | 444 |

图像尺寸 / Image sizes:

| group | size | count |
| --- | ---: | ---: |
| crack + normal | 640 x 640 | 337 |
| broken | 1920 x 1080 | 107 |

中文：

需要特别说明：`broken` 不是当前 U-Net 监督训练中的一个有标注类别。它在当前流程中是 unlabeled holdout，用于推理和定性检查。如果论文中讨论 `broken`，必须说明它没有进入 Stage1/Stage2 的监督训练和 CV 指标。

English:

It is important to state that `broken` is not a supervised labeled class in the current U-Net training pipeline. It is treated as unlabeled holdout data for inference and qualitative inspection. If `broken` is discussed in the thesis, it should be clearly described as excluded from Stage1/Stage2 supervised training and CV metrics.

## 3. Manifest and Mask Generation / 样本索引与 Mask 生成

入口 / Entry point:

```bash
python scripts/prepare_samples.py
```

默认参数 / Default split parameters:

```bash
python scripts/prepare_samples.py --test-ratio 0.20 --test-seed 2026
```

中文：

`prepare_samples.py` 是整个项目的数据入口。它扫描原始 ROI 数据集，给每张图像生成一行样本记录，并把 `crack` 图像的 LabelMe polygon 转换成二值 mask。输出的 `samples.csv` 是后续所有训练、验证和推理脚本的主索引。

English:

`prepare_samples.py` is the data entry point of the project. It scans the raw ROI dataset, creates one sample record for each image, and converts LabelMe polygons for `crack` images into binary masks. The output `samples.csv` is the central index used by all later training, validation, and inference scripts.

主要逻辑 / Main logic:

| step | 中文 | English |
| --- | --- | --- |
| scan | 遍历 `camera/phone` 下的 `crack/normal/broken`。 | Scan `crack/normal/broken` folders under `camera/phone`. |
| crack mask | 读取 LabelMe JSON 中的 polygon，用 PIL `ImageDraw.polygon` 填充为 255。 | Read LabelMe polygons and rasterize them into binary masks using PIL. |
| normal mask | `mask_path` 留空，Dataset 读取时生成全 0 mask。 | Keep `mask_path` empty; the Dataset creates an all-zero mask. |
| broken handling | 标记为 `sample_type=broken_unlabeled`，直接放入 holdout。 | Mark as `sample_type=broken_unlabeled` and place directly into holdout. |
| train/holdout split | 默认保留 20% `crack + normal` 为 inference-only holdout。 | Reserve 20% of `crack + normal` as inference-only holdout by default. |
| fold assignment | 对剩余 `trainval` 样本按 `(device, sample_type, defect_class)` 分组后分配 4 folds。 | Assign 4 folds to remaining `trainval` samples, stratified by `(device, sample_type, defect_class)`. |

输出 / Outputs:

```text
manifests/samples.csv
manifests/samples_summary.json
generated_masks/{device}/crack/*.png
```

`samples.csv` 关键字段 / Key fields in `samples.csv`:

| field | 中文 | English |
| --- | --- | --- |
| `sample_id` | 由 device、class、文件名组成的稳定样本 id。 | Stable sample identifier derived from device, class, and file stem. |
| `image_name` | 原始图像文件名。 | Original image filename. |
| `image_path` | 图像绝对路径。 | Absolute path to the image. |
| `mask_path` | crack mask 路径；normal/broken 可为空。 | Crack mask path; empty for normal/broken samples. |
| `json_path` | LabelMe JSON 路径。 | Path to the LabelMe JSON file. |
| `sample_type` | `defect`、`normal` 或 `broken_unlabeled`。 | `defect`, `normal`, or `broken_unlabeled`. |
| `is_labeled` | 是否有监督 mask。 | Whether a supervised mask exists. |
| `device` | `camera` 或 `phone`。 | `camera` or `phone`. |
| `defect_class` | 原始目录类别，如 `crack`、`normal`、`broken`。 | Original folder class, such as `crack`, `normal`, `broken`. |
| `split` | `trainval` 或 `holdout`。 | `trainval` or `holdout`. |
| `cv_fold` | 仅 `trainval` 样本有 fold id。 | Fold id for `trainval` samples only. |
| `holdout_reason` | `test_split`、`broken_unlabeled` 或空。 | `test_split`, `broken_unlabeled`, or empty. |

注意 / Note:

中文：

`samples.csv` 存的是绝对路径。因此本地生成的 manifest 不应直接复制到服务器使用；在服务器上训练前应重新运行 `prepare_samples.py`，使路径指向服务器数据目录。

English:

`samples.csv` stores absolute paths. A locally generated manifest should not be reused directly on the server. Before server-side training, `prepare_samples.py` should be rerun so paths point to the server dataset.

## 4. Split Design / 数据划分设计

中文：

项目有两层划分：第一层是 `trainval` 与 `holdout`，第二层是在 `trainval` 内做 4-fold cross-validation。

English:

The project uses two levels of data splitting: first into `trainval` and `holdout`, and then into 4-fold cross-validation inside `trainval`.

Train/holdout split:

| split | crack | normal | broken | total | role |
| --- | ---: | ---: | ---: | ---: | --- |
| `trainval` | 211 | 59 | 0 | 270 | Stage1/Stage2 training and CV validation |
| `holdout` | 52 | 15 | 107 | 174 | inference only |

Holdout composition:

| holdout reason | count | 中文 | English |
| --- | ---: | --- | --- |
| `test_split` | 67 | 从 crack/normal 中按 20% 保留。 | Held out from crack/normal using the 20% split. |
| `broken_unlabeled` | 107 | 全部 broken 图像。 | All broken images. |

Fold counts:

| fold | train images | train crack | train normal | val images | val crack | val normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 200 | 157 | 43 | 70 | 54 | 16 |
| 1 | 202 | 158 | 44 | 68 | 53 | 15 |
| 2 | 204 | 159 | 45 | 66 | 52 | 14 |
| 3 | 204 | 159 | 45 | 66 | 52 | 14 |

Fold rule:

```text
validation rows = split == trainval and cv_fold == current_fold
training rows   = split == trainval and cv_fold != current_fold
holdout rows    = split == holdout, never used for supervised training
```

中文：

当前 fold 划分是 image-level random split，不使用 video-level grouping。如果论文强调严格视频级泛化，需要额外说明当前版本没有使用视频映射文件。这是一个可在 limitations 中提到的点。

English:

The current fold assignment is an image-level random split and does not use video-level grouping. If strict video-level generalization is discussed, the thesis should state that the current version does not use video mapping files. This is a suitable limitation to report.

## 5. Crack Mask Statistics / 裂纹 Mask 统计

中文：

裂纹像素非常稀疏，这是使用 patch 预训练、`pos_weight`、Dice loss 和 normal replay 的主要原因。

English:

Crack pixels are highly sparse. This motivates patch-level pretraining, positive-pixel weighting, Dice loss, and normal replay.

Mask area over all 263 crack images:

| metric | mask area px | mask area % |
| --- | ---: | ---: |
| min | 76 | 0.0186 |
| p25 | 418 | 0.1021 |
| median | 1113 | 0.2717 |
| mean | 3590.7 | 0.8766 |
| p75 | 5225.5 | 1.2758 |
| p90 | 12304.8 | 3.0041 |
| max | 22165 | 5.4114 |

Device difference:

| source | count | median mask px | mean mask px | median mask % | mean mask % |
| --- | ---: | ---: | ---: | ---: | ---: |
| camera | 157 | 2911 | 5570.8 | 0.7107 | 1.3601 |
| phone | 106 | 382 | 658.0 | 0.0933 | 0.1606 |

中文：

phone 裂纹显著更小，median mask area 只有 0.0933%。这意味着模型容易被背景主导，也解释了为什么单纯整图训练会看到很少正像素。

English:

Phone cracks are substantially smaller, with a median mask area of only 0.0933%. This means the model can easily be dominated by background pixels, which explains why direct full-image training would expose the model to very few positive pixels.

## 6. Dataset Classes and Transforms / Dataset 读取与变换

核心文件 / Core file:

```text
src/datasets.py
```

中文：

数据读取统一通过 PIL 完成。图像转换为 RGB，mask 转为单通道二值图。训练前图像和 mask 都 resize 到目标尺寸，图像使用 bilinear interpolation，mask 使用 nearest-neighbor interpolation，避免 mask 边界被插值成灰度软标签。

English:

Images are loaded with PIL and converted to RGB. Masks are converted into single-channel binary arrays. Before training, both images and masks are resized to the target size. Images use bilinear interpolation, while masks use nearest-neighbor interpolation to avoid soft grayscale labels at mask boundaries.

Tensor format:

```text
image: float tensor [3, H, W], scaled to [0, 1]
mask:  float tensor [1, H, W], values in {0, 1}
```

Normalization:

```text
use_imagenet_normalize: true
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Augmentations:

| transform | Stage1 default | Stage2 default | 中文说明 | English |
| --- | ---: | ---: | --- | --- |
| horizontal flip | 0.5 | 0.5 | 图像和 mask 同步翻转。 | Synchronized image/mask flip. |
| vertical flip | 0.5 | 0.5 | 图像和 mask 同步翻转。 | Synchronized image/mask flip. |
| rotation | 15 deg | 10 deg | 图像 bilinear，mask nearest。 | Bilinear for images, nearest for masks. |
| brightness | 0.15 | 0.10 | 只作用于图像。 | Image only. |
| contrast | 0.15 | 0.10 | 只作用于图像。 | Image only. |
| gamma | 0.15 | 0.0 | 只作用于图像。 | Image only. |
| Gaussian noise | 0.02 | 0.015 | 只作用于图像。 | Image only. |
| Gaussian blur | 0.15 | 0.0 | 只作用于图像。 | Image only. |

PatchDataset worker-local cache:

中文：

`PatchDataset` 支持 worker-local cache。每个 DataLoader worker 内部维护 `_image_cache` 和 `_mask_cache`，缓存已读取的原图和 mask。`patch_worker_cache: true` 时启用；`patch_worker_cache_max_items: 0` 表示不设数量上限。该机制只减少重复 I/O，不改变样本顺序、增强随机性或训练目标。

English:

`PatchDataset` supports a worker-local cache. Each DataLoader worker maintains `_image_cache` and `_mask_cache` dictionaries for already loaded images and masks. It is enabled with `patch_worker_cache: true`; `patch_worker_cache_max_items: 0` means no item-count limit. This mechanism reduces repeated I/O only and does not change sample ordering, augmentation randomness, or the learning objective.

## 7. Stage1 Patch Index / Stage1 Patch 索引生成

入口 / Entry point:

```bash
python scripts/build_patch_index.py --config configs/stage1.yaml
```

中文：

Stage1 不直接训练整图，而是把每张 trainval 图像展开为多个 patch rows。每一行 patch 只保存裁剪窗口信息，不提前保存 patch 图片，因此不会额外复制大量图像数据。训练时 `PatchDataset` 根据 `image_path + crop_x + crop_y + crop_size` 动态裁剪。

English:

Stage1 does not train directly on full images. Instead, each trainval image is expanded into multiple patch rows. A patch row stores crop-window metadata rather than saving a cropped patch image, avoiding unnecessary image duplication. During training, `PatchDataset` dynamically crops the patch using `image_path + crop_x + crop_y + crop_size`.

Patch row fields:

```text
patch_id
base_sample_id
image_path
mask_path
patch_type
patch_family
crop_x
crop_y
crop_size
out_size
component_id
component_area_px
is_replay
replay_score
source_epoch
```

Patch generation settings:

| parameter | value |
| --- | ---: |
| `patch_out_size` | 384 |
| crop sizes | 320, 384, 448 |
| `max_components_per_image` | 3 |
| `max_positive_patches_per_image` | 6 |
| `patch_dedup_iou` | 0.70 |
| `near_miss_margin_min/max` | 8 / 20 |
| `hard_negative_safety_margin` | 24 |
| `max_attempts_per_patch` | 20 |

Patch types:

| patch type | family | 中文说明 | English description |
| --- | --- | --- | --- |
| `positive_center` | positive | 以缺陷 component 中心附近裁剪。 | Crop around the component center. |
| `positive_shift` | positive | 随机移动窗口，但保证包含缺陷 bbox。 | Shifted crop that still contains the defect bbox. |
| `positive_context` | positive | 使用更大上下文窗口包含缺陷。 | Larger contextual crop containing the defect. |
| `positive_boundary` | positive | 让缺陷靠近 patch 边缘，学习边界场景。 | Places the defect near a patch edge to learn boundary cases. |
| `near_miss_negative` | defect_negative | 在缺陷附近但不含缺陷。 | Near the defect but with no positive mask pixels. |
| `hard_negative` | defect_negative | 同一 defect 图上远离缺陷的背景。 | Background from a defect image, away from the defect. |
| `normal_negative` | normal_negative | normal 图像随机裁剪。 | Random crop from a normal image. |

中文：

`near_miss_negative` 和 `hard_negative` 的区别在于前者靠近缺陷边界，专门学习“像缺陷附近背景但不是缺陷”的区域；后者远离缺陷，用于学习缺陷图像里的普通背景。`normal_negative` 来自完全 normal 图，用于补充正常背景分布。

English:

`near_miss_negative` and `hard_negative` target different negative contexts. Near-miss negatives are close to defect boundaries and help the model learn background that resembles the surroundings of cracks. Hard negatives are farther away from defects within defect images. Normal negatives come from fully normal images and represent the normal background distribution.

Stage1 patch counts:

| fold | train patches | val patches | positive center | positive shift | positive context | positive boundary | near-miss negative | hard negative | normal negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1173 | 425 | 152 | 314 | 157 | 314 | 51 | 67 | 118 |
| 1 | 1194 | 401 | 154 | 316 | 158 | 316 | 59 | 66 | 125 |
| 2 | 1195 | 402 | 158 | 318 | 159 | 318 | 61 | 60 | 121 |
| 3 | 1231 | 361 | 154 | 318 | 159 | 318 | 62 | 79 | 141 |

## 8. Model Architecture / 模型结构

核心文件 / Core file:

```text
src/model.py
```

中文：

模型是 ResNet34 encoder + U-Net decoder。encoder 使用 torchvision 的 ResNet34，Stage1 默认尝试加载 ImageNet 预训练权重；Stage2 默认不再从 ImageNet 初始化，而是加载同 fold 的 Stage1 checkpoint。

English:

The model is a ResNet34 encoder with a U-Net decoder. The encoder is based on torchvision ResNet34. Stage1 attempts to use ImageNet-pretrained weights by default. Stage2 does not initialize from ImageNet; it loads the Stage1 checkpoint from the same fold.

Forward path:

```text
input image
  -> ResNet stem: conv1 + bn1 + relu
  -> maxpool + layer1
  -> layer2
  -> layer3
  -> layer4
  -> center ConvBlock
  -> decoder4 with skip from layer3
  -> decoder3 with skip from layer2
  -> decoder2 with skip from layer1
  -> decoder1 with skip from stem
  -> upsample to input size
  -> segmentation head
  -> 1-channel logits
```

Decoder details:

| module | mechanism |
| --- | --- |
| `ConvBlock` | two `3 x 3` conv layers, each with BatchNorm and ReLU |
| `DecoderBlock` | bilinear upsampling, skip concatenation, then `ConvBlock` |
| segmentation head | `3 x 3 conv -> BN -> ReLU -> 1 x 1 conv` |

中文：

模型输出的是 logits，不是 sigmoid 后的概率。训练损失直接使用 logits；验证和推理阶段才通过 sigmoid 得到 probability map。

English:

The model outputs logits rather than sigmoid probabilities. The training loss consumes logits directly. Sigmoid conversion is applied only during validation and inference to obtain probability maps.

Optional heads:

| option | 中文 | English |
| --- | --- | --- |
| deep supervision | 在 decoder 的 `d4/d3/d2` 上增加 auxiliary heads，并上采样到输入尺寸计算辅助 loss。 | Adds auxiliary heads on decoder features `d4/d3/d2`, upsampled to input size for auxiliary losses. |
| boundary auxiliary | 在最终 decoder feature `d1` 上增加 boundary head，预测 mask 边界。 | Adds a boundary head on final decoder feature `d1` to predict mask boundaries. |

Checkpoint compatibility:

中文：

deep supervision 或 boundary auxiliary 会新增 head，因此从普通 Stage1 checkpoint 加载时使用 `strict=False`。这允许共享 encoder/decoder/segmentation head 权重，同时让新增辅助 head 随机初始化。

English:

Deep supervision and boundary auxiliary settings introduce new heads. Therefore, when loading a plain Stage1 checkpoint, `strict=False` is used. This reuses the encoder, decoder, and segmentation head weights while allowing the new auxiliary heads to be randomly initialized.

## 9. Loss Functions / 损失函数机制

核心文件 / Core file:

```text
src/losses.py
```

Base segmentation loss:

```text
loss_seg = bce_weight * BCEWithLogitsLoss + dice_weight * DiceLoss

default:
bce_weight  = 0.5
dice_weight = 0.5
```

Dice loss:

```text
Dice = (2 * sum(prob * target) + eps) / (sum(prob) + sum(target) + eps)
DiceLoss = 1 - Dice
```

`pos_weight`:

中文：

`pos_weight` 是 BCEWithLogitsLoss 中正像素的权重。由于裂纹像素占比很低，普通 BCE 容易被背景像素主导。增大 `pos_weight` 可以提高正像素错误的代价，使模型更重视缺陷区域。本项目最终选择 `pos_weight=12`。

English:

`pos_weight` is the positive-pixel weight in BCEWithLogitsLoss. Because crack pixels occupy a very small fraction of each image, plain BCE can be dominated by background pixels. Increasing `pos_weight` makes positive-pixel errors more costly and forces the model to focus on defect regions. The selected setting is `pos_weight=12`.

Normal false-positive loss:

```text
For empty-target samples:
  normal_probs = sigmoid(logits)
  normal_fp_loss = mean(top-k normal probabilities)
  total_loss += normal_fp_loss_weight * normal_fp_loss

default tested:
normal_fp_topk_ratio = 0.10
normal_fp_loss_weight in {0.03, 0.05}
```

中文：

这个 loss 只对 target 为空的样本生效，目的是惩罚 normal 图中最高概率的一小部分像素。但实验显示它没有改善最终结果：`0.03` 增加 normal false positives，`0.05` 降低 Dice 和 recall。

English:

This loss is applied only to samples with empty targets. It penalizes the highest-probability pixels on normal images. However, experiments showed no improvement: `0.03` increased normal false positives, while `0.05` reduced Dice and recall.

Deep supervision loss:

```text
aux_loss = weighted average of segmentation losses from aux heads
weights  = deep_supervision_decay ** index
total_loss += deep_supervision_weight * aux_loss

tested:
deep_supervision_weight = 0.20
deep_supervision_decay  = 0.50
```

中文：

辅助 head 的 loss 不包含 normal false-positive loss，只使用 segmentation loss。这样做可以避免多个尺度的辅助输出同时过度惩罚 normal 图。

English:

The auxiliary head losses use only the segmentation loss and do not include the normal false-positive term. This avoids over-penalizing normal images across multiple auxiliary scales.

Boundary auxiliary loss:

```text
boundary_target = dilated(mask) - eroded(mask)
boundary_loss   = BCEWithLogitsLoss(boundary_logits, boundary_target)
total_loss += boundary_aux_weight * boundary_loss

tested:
boundary_aux_weight = 0.10
boundary_width      = 3
```

中文：

boundary target 由 max-pooling dilation 和 erosion 构造，表示 mask 边界区域。该机制希望改善细长裂纹边界，但实验中没有超过 baseline，并且增加了 normal false positives。

English:

The boundary target is constructed using max-pooling dilation and erosion and represents the mask boundary region. The purpose is to improve thin crack boundaries, but it did not outperform the baseline and increased normal false positives.

## 10. Stage1 Training / Stage1 训练机制

入口 / Entry point:

```bash
python scripts/train_stage1.py --config configs/stage1.yaml --fold 0
```

Default P0 A40 Stage1 config:

```text
configs/stage1_p0_a40.yaml
patch_out_size: 384
batch_size: 128
epochs: 30
pretrained: true
freeze_encoder_epochs: 3
stage1_sampler_mode: balanced
stage1_positive_ratio: 0.50
stage1_defect_negative_ratio: 0.25
stage1_normal_ratio: 0.25
stage1_use_replay: true
stage1_replay_ratio: 0.15
```

Stage1 data flow:

```text
stage1_fold{fold}_train_index.csv
  -> PatchDataset
  -> dynamic crop from original image and mask
  -> resize to 384 x 384
  -> train augmentation
  -> ResNet34-U-Net
  -> patch logits
  -> BCE + Dice loss
```

Balanced sampler:

中文：

Stage1 使用 `WeightedRandomSampler` 让训练 batch 更接近指定 family ratio，而不是完全按 patch index 的自然比例采样。默认目标为 positive 50%、defect negative 25%、normal negative 25%。这能缓解正负 patch 不均衡，也让模型更频繁看到正样本。

English:

Stage1 uses a `WeightedRandomSampler` to make training batches closer to the configured family ratios rather than the raw patch-index distribution. The default target is 50% positive patches, 25% defect negatives, and 25% normal negatives. This reduces patch-level imbalance and exposes the model to positives more frequently.

Encoder freezing:

中文：

Stage1 前 3 个 epoch 冻结 encoder 参数，只训练 decoder 和 segmentation head。冻结时会调用 `apply_encoder_freeze_mode()`，把 encoder modules 设置为 eval mode，从而冻结 BatchNorm running statistics，避免 frozen encoder 的 BN 均值方差仍被训练数据更新。

English:

During the first 3 Stage1 epochs, encoder parameters are frozen and only the decoder and segmentation head are trained. When the encoder is frozen, `apply_encoder_freeze_mode()` sets encoder modules to evaluation mode, which freezes BatchNorm running statistics and prevents them from being updated by training batches.

Stage1 validation metrics:

| metric | 中文 | English |
| --- | --- | --- |
| `patch_dice_all` | 所有 patch 的平均 Dice。 | Mean Dice over all patches. |
| `patch_dice_pos_only` | positive patch 的平均 Dice。 | Mean Dice over positive patches. |
| `positive_patch_recall` | positive patch 是否预测出任意正像素。 | Fraction of positive patches with any positive prediction. |
| `negative_patch_fpr` | negative patch 中预测出正像素的比例。 | Fraction of negative patches with any positive prediction. |

Stage1 checkpoint score:

```text
stage1_score =
  patch_dice_pos_only
  - stage1_negative_fpr_penalty
    * max(0, negative_patch_fpr - stage1_target_negative_fpr)

default:
stage1_target_negative_fpr  = 0.20
stage1_negative_fpr_penalty = 0.5
```

中文：

Stage1 选 checkpoint 时不只追求正样本 Dice，还惩罚过高的 negative patch FPR。这样 Stage1 权重不会变成只会“到处找裂纹”的高召回但高误检模型。

English:

Stage1 checkpoint selection does not optimize positive Dice alone. It also penalizes excessive negative patch FPR. This prevents Stage1 from becoming a high-recall but high-false-positive model.

Stage1 outputs:

```text
outputs/.../stage1/fold{fold}/best_stage1.pt
outputs/.../stage1/fold{fold}/last_stage1.pt
outputs/.../stage1/fold{fold}/history.csv
outputs/.../stage1/fold{fold}/replay/
```

## 11. Stage1 Replay / Stage1 困难样本回放

中文：

Stage1 replay 周期性扫描 base patch index，找当前模型预测困难的 patch，然后把这些 patch 作为 replay rows 加入后续 epoch 的训练。这样模型可以反复看到 false negative 和 false positive，而不是每轮只依赖固定 patch index。

English:

Stage1 replay periodically scans the base patch index, finds patches that are difficult for the current model, and appends them as replay rows in later epochs. This makes the model revisit false negatives and false positives instead of relying only on the fixed patch index.

Replay schedule:

```text
stage1_replay_warmup_epochs: 3
stage1_replay_refresh_every: 3
stage1_replay_ratio: 0.15
stage1_max_replay_ratio: 0.20
stage1_eval_threshold: 0.50
```

Replay data flow:

```text
base patch rows
  -> current Stage1 model prediction
  -> sigmoid probability map
  -> threshold at 0.50
  -> score difficult positives and negatives
  -> select top replay rows
  -> append to next training epochs
```

Replay types:

| replay type | 中文 | English |
| --- | --- | --- |
| `replay_positive_fn` | positive patch 没有预测出正像素。 | Positive patch with no positive prediction. |
| `replay_positive_hard` | positive patch 预测不够好但不是完全漏检。 | Hard positive patch with imperfect prediction. |
| `replay_defect_negative_fp` | defect negative patch 上出现误检。 | False positive on a defect-negative patch. |
| `replay_normal_negative_fp` | normal negative patch 上出现误检。 | False positive on a normal-negative patch. |

Replay scoring:

```text
positive score = 1 - Dice
if positive patch is completely missed:
  score += 1

negative score = predicted_positive_ratio + max_probability
```

Outputs:

```text
outputs/.../stage1/fold{fold}/replay/replay_latest.csv
outputs/.../stage1/fold{fold}/replay/replay_epochXXX.csv
outputs/.../stage1/fold{fold}/replay/replay_latest_summary.json
```

## 12. Stage2 Full-Image Training / Stage2 整图训练

入口 / Entry point:

```bash
python scripts/train_stage2.py --config configs/stage2.yaml --fold 0
```

Selected P0 A40 Stage2 baseline:

```text
configs/stage2_p0_a40_e50_bs48_pw12.yaml
image_size: 640
batch_size: 48
epochs: 50
pos_weight: 12.0
normal_fp_loss_weight: 0.0
random_normal_k_factor: 1.0
use_hard_normal_replay: true
stage2_hard_normal_ratio: 0.40
hard_normal_max_repeats_per_epoch: 2
threshold_grid_end: 0.95
```

中文：

Stage2 输入单位是完整 ROI 图像，而不是 patch。每个 fold 的 Stage2 会加载同 fold 的 Stage1 `best_stage1.pt`，继续训练同一个 U-Net 架构。这样 Stage1 学到的局部缺陷特征会迁移到整图训练中。

English:

Stage2 operates on full ROI images rather than patches. For each fold, Stage2 loads the corresponding Stage1 `best_stage1.pt` and continues training the same U-Net architecture. Thus, local defect features learned in Stage1 are transferred to full-image training.

Stage2 data flow:

```text
samples.csv
  -> split_samples_for_fold()
  -> defect_train_rows / defect_val_rows
  -> normal_train_rows / normal_val_rows
  -> ROIDataset
  -> full 640 x 640 image + mask
  -> Stage1-initialized U-Net
  -> full-image logits
  -> BCE + Dice loss
```

Per-epoch training rows:

```text
epoch_train_rows =
  all defect_train_rows
  + sampled random normal rows
  + sampled hard normal rows
```

Normal budget:

```text
normal_budget_count = round(defect_train_count * random_normal_k_factor)

default:
random_normal_k_factor = 1.0
```

中文：

也就是说每个 epoch 使用全部 defect training images，同时采样约等于 defect 数量的 normal images。这样整图训练中 defect 和 normal 图像数量大致平衡。

English:

Each epoch uses all defect training images and samples approximately the same number of normal images as the number of defect images. This keeps full-image Stage2 training roughly balanced at the image level.

Stage2 outputs:

```text
outputs/.../stage2/fold{fold}/best_stage2.pt
outputs/.../stage2/fold{fold}/last_stage2.pt
outputs/.../stage2/fold{fold}/history.csv
outputs/.../stage2/fold{fold}/hard_normal/
outputs/.../stage2/fold{fold}/val_metrics.json
outputs/.../stage2/fold{fold}/val_per_image.csv
outputs/.../stage2/fold{fold}/val_postprocess_search.csv
```

## 13. Stage2 Hard Normal Replay / Stage2 Hard Normal 回放

中文：

Stage2 hard normal replay 用于压低 normal 图上的误检。训练过程中模型会周期性扫描当前 fold 的 normal training images。如果某些 normal 图在当前模型下产生了后处理后的正像素，这些图会被加入 hard normal pool，并在后续 epoch 中替换一部分随机 normal 样本。

English:

Stage2 hard normal replay is designed to reduce false positives on normal images. During training, the model periodically scans normal training images from the current fold. If a normal image produces positive pixels after post-processing, it is added to the hard-normal pool and later replaces part of the random normal samples in subsequent epochs.

Schedule:

```text
hard_normal_warmup_epochs: 2
hard_normal_refresh_every: 2
stage2_hard_normal_ratio: 0.40
hard_normal_pool_factor: 3.0
hard_normal_max_repeats_per_epoch: 2
```

Hard-normal mining flow:

```text
normal_train_rows
  -> current Stage2 model
  -> sigmoid probability map
  -> threshold + min_area post-processing
  -> false-positive mask on normal images
  -> score by component size and probability
  -> ranked hard-normal pool
```

Hard score:

```text
hard_score =
  largest_fp_component_area * 1,000,000
  + fp_pixel_count
  + max_probability
```

中文：

评分优先看最大误检连通域面积，其次看误检像素总数，最后看最大概率。这使大块误检比零散噪声优先被 replay。

English:

The score prioritizes the largest false-positive connected component, then total false-positive pixels, and finally the maximum probability. This makes large false-positive regions more likely to be replayed than isolated noise.

Hard-normal sampling cap:

中文：

早期版本的问题是：如果 hard pool 很小，但配置要求很多 hard normals，同一张 normal 图可能被重复采样几十次。现在 `hard_normal_max_repeats_per_epoch=2` 限制每个 hard-normal row 在单个 epoch 中最多重复 2 次。实际 hard-normal 数量会被截断为：

English:

An earlier issue was that a small hard pool could cause the same normal image to be repeated many times within an epoch. The current `hard_normal_max_repeats_per_epoch=2` limits each hard-normal row to at most 2 repetitions per epoch. The actual hard-normal count is capped as:

```text
target_hard_normal_count =
  min(requested_hard_normal_count,
      len(hard_normal_pool) * hard_normal_max_repeats_per_epoch)
```

Outputs:

```text
outputs/.../stage2/fold{fold}/hard_normal/hard_normal_latest.csv
outputs/.../stage2/fold{fold}/hard_normal/hard_normal_epochXXX.csv
outputs/.../stage2/fold{fold}/hard_normal/hard_normal_latest_summary.json
```

## 14. Stage2 Validation and Checkpoint Selection / Stage2 验证与模型选择

中文：

Stage2 每个 epoch 都在当前 fold 的 validation set 上评估。重要改动是：训练过程中的 checkpoint 选择使用固定 `train_eval_threshold/train_eval_min_area`，不再每个 epoch 搜索 threshold 和 min_area。这样 checkpoint 选择更稳定，也避免训练期间过度依赖验证集后处理搜索。

English:

Stage2 is evaluated on the current fold validation set after each epoch. A key stabilization change is that checkpoint selection during training uses fixed `train_eval_threshold/train_eval_min_area`, rather than searching threshold and min_area every epoch. This makes checkpoint selection more stable and avoids excessive validation-set post-processing tuning during training.

Train-time selection parameters:

```text
train_eval_threshold: 0.50
train_eval_min_area: 0
```

Validation flow:

```text
validation images
  -> model logits
  -> sigmoid probability maps
  -> fixed threshold/min_area during training
  -> binary masks
  -> per-image metrics
  -> fold-level metrics
  -> checkpoint comparison
```

Metrics:

| metric | 中文 | English |
| --- | --- | --- |
| `defect_dice` | 只在 defect validation images 上平均 Dice。 | Mean Dice over defect validation images only. |
| `defect_iou` | 只在 defect validation images 上平均 IoU。 | Mean IoU over defect validation images only. |
| `defect_image_recall` | defect 图是否预测出任意正像素的比例。 | Fraction of defect images with any positive prediction. |
| `normal_fpr` | normal 图中出现任意正预测的图像比例。 | Fraction of normal images with any positive prediction. |
| `normal_fp_count` | 出现误检的 normal 图数量。 | Number of normal images with false positives. |
| `normal_fp_pixel_*` | normal 图误检像素统计。 | Pixel-level false-positive statistics on normal images. |
| `normal_largest_fp_area_*` | normal 图最大误检连通域统计。 | Largest false-positive connected-component statistics. |

Stage2 score:

```text
stage2_score =
  defect_dice
  - lambda_fpr_penalty * max(0, normal_fpr - target_normal_fpr)

default:
target_normal_fpr = 0.10
lambda_fpr_penalty = 2.0
```

Checkpoint comparison:

```text
1. Prefer models with normal_fpr <= target_normal_fpr.
2. Among acceptable models, prefer higher defect_dice.
3. If defect_dice ties, prefer lower normal_fpr.
4. If both tie, prefer higher defect_image_recall.
5. If no model satisfies target_normal_fpr, compare stage2_score.
```

中文：

这个选择规则体现了任务目标：先保证 normal 误检率不要失控，然后最大化缺陷分割质量。

English:

This selection rule reflects the task objective: first keep the normal false-positive rate under control, then maximize defect segmentation quality.

## 15. Post-Processing / 后处理机制

核心文件 / Core file:

```text
src/metrics.py
```

中文：

模型输出经过 sigmoid 得到 probability map。后处理先按 threshold 二值化，再用 8-neighborhood connected components 删除面积小于 `min_area` 的小连通域。

English:

Model logits are converted to probability maps using sigmoid. Post-processing first thresholds the probability map and then removes connected components smaller than `min_area` using 8-neighborhood connected components.

Post-processing formula:

```text
prob_map = sigmoid(logits)
raw_mask = prob_map >= threshold
final_mask = remove_connected_components(raw_mask, area < min_area)
```

P0 A40 search grid:

```text
threshold_grid_start: 0.10
threshold_grid_end: 0.95
threshold_grid_step: 0.02
min_area_grid: [0, 8, 16, 24, 32, 48]
```

中文：

`threshold` 控制像素级置信度门槛，`min_area` 控制是否删除小噪声连通域。裂纹很细，因此 `min_area` 不能过大，否则可能误删真实小裂纹；但 `min_area` 太小又会保留 normal 图上的噪声误检。

English:

`threshold` controls the pixel-level confidence cutoff, while `min_area` controls removal of small noisy components. Since cracks can be very thin, `min_area` cannot be too large or it may remove real small cracks. If it is too small, noisy false positives on normal images may remain.

## 16. Per-Fold Validation Export / 单 Fold 验证导出

入口 / Entry point:

```bash
python scripts/evaluate_val.py --config configs/stage2.yaml --fold 0
```

中文：

`train_stage2.py` 训练结束后默认自动调用 validation export，因为配置中 `auto_evaluate_after_train: true`。该步骤会同时导出 raw threshold=0.5 的结果和后处理搜索结果。

English:

`train_stage2.py` automatically runs validation export after training because `auto_evaluate_after_train: true` is set in the config. This step exports both raw threshold=0.5 results and post-processing search results.

Outputs:

```text
outputs/.../stage2/fold{fold}/val_metrics.json
outputs/.../stage2/fold{fold}/val_per_image.csv
outputs/.../stage2/fold{fold}/val_postprocess_search.csv
```

`val_metrics.json`:

中文：

保存该 fold 的最佳后处理参数和 summary metrics，包括 raw metrics、post-processed metrics、threshold、min_area、target_normal_fpr 和 checkpoint 内部记录的最佳参数。

English:

Stores the best post-processing parameters and summary metrics for the fold, including raw metrics, post-processed metrics, threshold, min_area, target_normal_fpr, and the checkpoint's internal best parameters.

`val_per_image.csv`:

中文：

逐图保存 Dice、IoU、是否有正预测、误检像素数、最大误检连通域面积。适合做 error analysis。

English:

Stores per-image Dice, IoU, positive-prediction flag, false-positive pixel count, and largest false-positive component area. It is useful for error analysis.

`val_postprocess_search.csv`:

中文：

保存所有 threshold/min_area 组合的搜索结果，可用于画后处理参数敏感性曲线。

English:

Stores all threshold/min_area search results and can be used to plot post-processing sensitivity curves.

## 17. Pooled OOF Global Search / Pooled OOF 全局后处理搜索

入口 / Entry point:

```bash
python scripts/search_oof_postprocess.py --config configs/stage2.yaml --folds 0,1,2,3
```

中文：

OOF 是 out-of-fold。对每一张 trainval 图像，只用没有训练过它的 fold 模型做一次预测。然后把四个 fold 的 validation predictions 合并，在合并集合上搜索一个统一的 `threshold/min_area`。

English:

OOF means out-of-fold. Each trainval image is predicted once by the fold model that did not train on that image. Predictions from all four validation folds are pooled, and a single global `threshold/min_area` pair is selected on the pooled set.

OOF flow:

```text
fold0 best_stage2.pt -> predict fold0 validation rows
fold1 best_stage2.pt -> predict fold1 validation rows
fold2 best_stage2.pt -> predict fold2 validation rows
fold3 best_stage2.pt -> predict fold3 validation rows
  -> pool all validation predictions
  -> search global threshold/min_area
  -> save global metrics and per-image table
```

Outputs:

```text
outputs/.../stage2/oof_global_postprocess.json
outputs/.../stage2/oof_global_postprocess_search.csv
outputs/.../stage2/oof_per_image.csv
```

中文：

论文中建议报告 pooled OOF 结果，而不是单个 fold 的最优结果。pooled OOF 更接近交叉验证整体表现，也能避免挑选某个 fold 的偶然优势。

English:

For the thesis, pooled OOF results are recommended instead of the best single-fold result. Pooled OOF better represents the overall cross-validation performance and avoids cherry-picking a favorable fold.

## 18. Holdout Inference / Holdout 推理

入口 / Entry point:

```bash
python scripts/infer_holdout.py --config configs/stage2.yaml --fold 0
```

中文：

holdout 推理读取 `samples.csv` 中 `split == holdout` 的样本。它优先使用 `global_postprocess_path` 指向的 pooled OOF 后处理参数。如果 global 文件不存在，则回退到当前 fold 的 `val_metrics.json`。

English:

Holdout inference reads samples with `split == holdout` from `samples.csv`. It first tries to use the pooled OOF post-processing parameters from `global_postprocess_path`. If that file does not exist, it falls back to the current fold's `val_metrics.json`.

Parameter priority:

```text
1. global_postprocess_path / oof_global_postprocess.json
2. fold-level val_metrics.json
```

Holdout outputs:

```text
outputs/.../stage2/fold{fold}/holdout/prob_maps/
outputs/.../stage2/fold{fold}/holdout/raw_binary_masks/
outputs/.../stage2/fold{fold}/holdout/masks/
outputs/.../stage2/fold{fold}/holdout/inference_summary.csv
```

中文：

`holdout` 里的 crack 和 normal 有监督含义，可以作为最终保留测试集进一步评估；`broken` 没有 GT mask，因此只能做定性预测检查，不能进入 Dice/IoU 等监督指标。

English:

The crack and normal samples in `holdout` can be used as a final held-out test set if evaluated with their labels. The `broken` samples do not have GT masks, so they can only be used for qualitative prediction inspection, not supervised Dice/IoU metrics.

## 19. Error Analysis Visualization / 错误分析可视化

入口 / Entry point:

```bash
python scripts/visualize_error_analysis.py --config configs/stage2.yaml --fold 0
```

中文：

该脚本用于生成验证集错误分析图。它读取 best Stage2 checkpoint 和 fold validation rows，生成原图、预测/GT overlay 和 probability heatmap 拼接图，并按 false positive、false negative、worst defects 分组输出 contact sheet。

English:

This script generates validation error-analysis visualizations. It loads the best Stage2 checkpoint and fold validation rows, then renders panels containing the original image, prediction/GT overlay, and probability heatmap. It groups outputs into false positives, false negatives, and worst defects and creates contact sheets.

Overlay colors:

| color | meaning |
| --- | --- |
| green | ground-truth only |
| red | prediction only |
| yellow | prediction and ground-truth overlap |

Outputs:

```text
outputs/.../stage2/fold{fold}/error_analysis/summary.csv
outputs/.../stage2/fold{fold}/error_analysis/overlays/*.png
outputs/.../stage2/fold{fold}/error_analysis/false_positives_contact.png
outputs/.../stage2/fold{fold}/error_analysis/false_negatives_contact.png
outputs/.../stage2/fold{fold}/error_analysis/worst_defects_contact.png
outputs/.../stage2/fold{fold}/error_analysis/run_info.csv
```

论文用途 / Thesis use:

中文：

这些图可以用于 qualitative results 或 failure case analysis，展示模型在哪些裂纹上漏检、在哪些 normal 图上误检，以及后处理对小连通域噪声的影响。

English:

These visualizations can be used for qualitative results or failure-case analysis, showing which cracks are missed, which normal images produce false positives, and how post-processing affects small noisy components.

## 20. Multi-GPU Execution / 多 GPU 执行方式

中文：

当前 U-Net 多 GPU 不是 DDP，而是 fold-level parallelism。四张 GPU 分别跑四个 fold，每个 fold 内部仍是单进程单 GPU 训练。

English:

The current U-Net multi-GPU strategy is not DDP. It is fold-level parallelism: four GPUs run four folds independently, while each fold is trained by a single process on one GPU.

P0 A40 pattern:

```text
fold0 -> GPU0
fold1 -> GPU1
fold2 -> GPU2
fold3 -> GPU3
```

Typical commands:

```bash
bash scripts/run_p0_a40_parallel.sh
bash scripts/run_p0_a40_second_batch.sh
bash scripts/run_p0_a40_normal_fp_loss_ablation.sh
bash scripts/run_p0_a40_third_batch.sh
```

中文：

这种方式的优点是实现简单，fold 之间互不影响，且天然适合交叉验证。缺点是单个 fold 不能使用多卡加速。

English:

This approach is simple, keeps folds independent, and naturally fits cross-validation. The limitation is that a single fold is not accelerated by multiple GPUs.

## 21. Reproducibility / 可复现性机制

中文：

项目通过固定 Python、NumPy、PyTorch 随机种子，以及设置 cuDNN deterministic mode 增强可复现性。DataLoader worker 会通过 `seed_worker()` 基于 PyTorch worker seed 设置 Python 和 NumPy seed。

English:

The project improves reproducibility by fixing Python, NumPy, and PyTorch random seeds and enabling deterministic cuDNN behavior. DataLoader workers use `seed_worker()` to initialize Python and NumPy seeds from the PyTorch worker seed.

Mechanisms:

```text
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Stage1 per-epoch loader seed:

中文：

Stage1 的训练 loader 每个 epoch 使用不同 seed：`seed + epoch * 1009`。这样在线增强不会每个 epoch 以完全相同的随机序列重复，降低增强模式重复的问题。

English:

The Stage1 training loader uses a different seed each epoch: `seed + epoch * 1009`. This prevents online augmentations from repeating exactly the same random sequence across epochs.

## 22. Experiment Summary / 实验总结

中文：

2026-05-09 的主要 U-Net 实验分三批：`pos_weight` 消融、normal false-positive loss 消融、以及 deep supervision/boundary auxiliary/worker cache 第三批实验。完整日志已复制到 GitHub 的 `experiment_logs/20260509/unet/` 下；服务器原始输出仍保留在 `outputs/experiments/` 下。

English:

The main U-Net experiments on 2026-05-09 were organized into three batches: `pos_weight` ablation, normal false-positive loss ablation, and the third batch covering deep supervision, boundary auxiliary loss, and worker cache. Lightweight logs were copied into `experiment_logs/20260509/unet/` in GitHub, while original server outputs remain under `outputs/experiments/`.

Copied review folders:

```text
experiment_logs/20260509/unet/pos_weight_ablation_20260509_review/
experiment_logs/20260509/unet/normal_fp_loss_ablation_20260509_review/
experiment_logs/20260509/unet/third_batch_segmentation_20260509_review/
```

### 22.1 Pos Weight Ablation / `pos_weight` 消融

Purpose:

中文：

测试 BCE 中正像素权重对稀疏裂纹分割的影响。

English:

Test the effect of positive-pixel weighting in BCE for sparse crack segmentation.

| variant | pos_weight | Dice | IoU | defect recall | normal FPR | threshold | min_area | normal FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pw6 | 6 | 0.755910 | 0.635336 | 0.990521 | 0.016949 | 0.74 | 8 | 1 |
| pw8 | 8 | 0.751339 | 0.631859 | 0.981043 | 0.016949 | 0.76 | 24 | 1 |
| pw12 | 12 | 0.756876 | 0.633878 | 0.995261 | 0.000000 | 0.80 | 24 | 0 |

Conclusion:

中文：

`pos_weight=12` 最均衡，Dice 最高，defect recall 最高，并且 normal FPR 为 0。建议作为主结果写入论文。

English:

`pos_weight=12` is the best balanced setting. It gives the highest Dice, the highest defect recall, and zero normal FPR. It should be used as the main U-Net result in the thesis.

### 22.2 Normal FP Loss Ablation / Normal False-Positive Loss 消融

Purpose:

中文：

在 `pos_weight=12` 稳定后，测试额外 normal false-positive penalty 是否能进一步降低 normal 图误检。

English:

After stabilizing `pos_weight=12`, test whether an additional normal false-positive penalty can further reduce false positives on normal images.

| variant | normal_fp_loss_weight | Dice | IoU | defect recall | normal FPR | normal FP | threshold/min_area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline pw12 | 0 | 0.756876 | 0.633878 | 0.995261 | 0.000000 | 0 | 0.80/24 |
| pw12_fp003 | 0.03 | 0.741675 | 0.623785 | 0.966825 | 0.067797 | 4 | 0.82/8 |
| pw12_fp005 | 0.05 | 0.710942 | 0.589482 | 0.947867 | 0.000000 | 0 | 0.82/16 |

Conclusion:

中文：

normal FP loss 没有改善 baseline。`0.03` 反而产生更多 normal false positives，`0.05` 虽然保持 normal FPR 为 0，但 Dice 和 recall 明显下降。因此不推荐作为最终方法，但可以作为 negative ablation 写进论文。

English:

The normal FP loss did not improve the baseline. `0.03` introduced more normal false positives, while `0.05` preserved zero normal FPR but substantially reduced Dice and recall. It is not recommended as the final method, but it is useful as a negative ablation in the thesis.

### 22.3 Third-Batch Segmentation Experiments / 第三批分割实验

Purpose:

中文：

测试数据加载优化和两个辅助训练机制：PatchDataset worker-local cache、deep supervision、boundary auxiliary loss。

English:

Test data-loading optimization and two auxiliary training mechanisms: PatchDataset worker-local cache, deep supervision, and boundary auxiliary loss.

| variant | Dice | IoU | defect recall | normal FPR | normal FP | threshold | min_area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pw12 baseline | 0.756876 | 0.633878 | 0.995261 | 0.000000 | 0 | 0.80 | 24 |
| deep supervision | 0.752240 | 0.637742 | 0.957346 | 0.016949 | 1 | 0.86 | 48 |
| boundary auxiliary | 0.755884 | 0.633033 | 0.995261 | 0.033898 | 2 | 0.82 | 0 |

Conclusion:

中文：

deep supervision 略微提高 IoU，但 Dice 和 defect recall 下降，并出现 1 个 normal false positive。boundary auxiliary 没有提升 Dice/IoU，并增加 normal false positives。PatchDataset cache 是运行效率优化，不改变模型指标。因此 Stage2 `pos_weight=12` baseline 仍是最优 U-Net 方案。

English:

Deep supervision slightly improved IoU, but reduced Dice and defect recall and introduced one normal false positive. Boundary auxiliary loss did not improve Dice/IoU and increased normal false positives. PatchDataset cache is a runtime optimization and does not change metrics. Therefore, the Stage2 `pos_weight=12` baseline remains the best U-Net setting.

## 23. Logs and Artifact Map / 日志与产物位置

Root experiment summary:

```text
EXPERIMENTS_20260509.md
```

Dataset summary:

```text
DATASET_DESCRIPTION.md
```

This data-flow document:

```text
UNET_DATA_FLOW.md
```

Lightweight U-Net logs in GitHub:

```text
experiment_logs/20260509/unet/pos_weight_ablation_20260509_review/
experiment_logs/20260509/unet/normal_fp_loss_ablation_20260509_review/
experiment_logs/20260509/unet/third_batch_segmentation_20260509_review/
```

Important files inside review folders:

```text
README.md
configs/*.yaml
logs/*.log
stage2/fold*/history.csv
stage2/fold*/val_metrics.json
stage2/fold*/val_per_image.csv
stage2/fold*/val_postprocess_search.csv
stage2/oof_global_postprocess.json
stage2/oof_global_postprocess_search.csv
stage2/oof_per_image.csv
```

Server original outputs:

```text
outputs/experiments/p0_a40_20260508/
outputs/experiments/p0_a40_e50_bs48_pw6_20260509/
outputs/experiments/p0_a40_e50_bs48_pw8_20260509/
outputs/experiments/p0_a40_e50_bs48_pw12_20260509/
outputs/experiments/p0_a40_e50_bs48_pw12_fp003_20260509/
outputs/experiments/p0_a40_e50_bs48_pw12_fp005_20260509/
outputs/experiments/p0_a40_e50_bs48_pw12_deepsup_20260509/
outputs/experiments/p0_a40_e50_bs48_pw12_boundary_20260509/
outputs/experiments/p0_a40_patchdataset_cache_20260509/
```

中文：

GitHub 中只保留轻量文本和表格产物：配置、脚本、日志、CSV、JSON、README。模型 checkpoint、预测图像和大量可视化文件没有全部推到 GitHub，以避免仓库过大。

English:

Only lightweight text and table artifacts were kept in GitHub: configs, scripts, logs, CSV files, JSON files, and README files. Model checkpoints, prediction images, and large visualization outputs were not fully pushed to GitHub to avoid making the repository too large.

## 24. Thesis-Ready Method Description / 可用于论文的方法描述

中文方法描述：

本文提出并实现了一个面向接触线 ROI 图像缺陷分割的两阶段 U-Net 流程。原始数据首先被整理为统一的样本索引，其中有标注的 crack 图像由 LabelMe polygon 转换为二值 mask，normal 图像被视为全 0 mask 负样本，未标注的 broken 图像仅用于 holdout 推理。训练数据采用 image-level 4-fold cross-validation，并额外保留 20% crack/normal 样本作为 inference-only holdout。

第一阶段在局部 patch 上训练 ResNet34-U-Net。Patch index 包含正样本 patch、缺陷邻近负样本、缺陷图背景负样本和 normal 负样本。该设计提高了稀疏裂纹像素在训练中的出现频率，并使模型学习裂纹边界附近容易混淆的背景区域。Stage1 使用 BCE-Dice loss、正负 patch 平衡采样、encoder warm-up freeze 以及困难 patch replay。

第二阶段将同一 fold 的 Stage1 最优权重迁移到完整 640 x 640 ROI 图像上继续训练。每个 epoch 使用全部 defect training images，并按 defect 数量采样 normal images。训练过程中周期性挖掘 normal 图上的 false positives，形成 hard-normal pool，用于后续 epoch 替换部分随机 normal 样本。为了避免小 hard pool 导致重复采样过多，每个 hard-normal row 在单个 epoch 内最多重复 2 次。

验证阶段将 logits 转换为 probability map，经 threshold 和 connected-component area filtering 得到 binary mask。模型选择时使用固定 train-time threshold/min_area，最终后处理参数在 pooled out-of-fold validation predictions 上统一搜索。该策略使 checkpoint selection 和 final threshold selection 分离，减少每个 epoch 或每个 fold 单独调阈值带来的不稳定。

English method description:

This work implements a two-stage U-Net pipeline for defect segmentation in contact-wire ROI images. The raw dataset is first converted into a unified sample manifest. Labeled crack images are converted from LabelMe polygons to binary masks, normal images are treated as negative samples with all-zero masks, and unlabeled broken images are used only for holdout inference. The training data are split using image-level 4-fold cross-validation, with an additional 20% crack/normal subset reserved as inference-only holdout.

In Stage1, a ResNet34-U-Net is trained on local patches. The patch index contains positive patches, near-defect negative patches, background negatives from defect images, and negatives from normal images. This design increases the frequency of sparse crack pixels during training and exposes the model to confusing background regions near crack boundaries. Stage1 uses BCE-Dice loss, balanced patch-family sampling, encoder warm-up freezing, and hard patch replay.

In Stage2, the best Stage1 checkpoint from the same fold is transferred to full 640 x 640 ROI images. Each epoch uses all defect training images and samples normal images according to the number of defect images. During training, false positives on normal images are periodically mined into a hard-normal pool, which replaces part of the random normal samples in later epochs. To avoid excessive repetition when the hard pool is small, each hard-normal row is repeated at most twice per epoch.

During validation, logits are converted to probability maps and then post-processed with thresholding and connected-component area filtering. Checkpoint selection uses fixed train-time threshold/min_area values, while final post-processing parameters are selected by a global search over pooled out-of-fold validation predictions. This separates checkpoint selection from final threshold tuning and reduces instability from epoch-specific or fold-specific threshold search.

## 25. What Can Be Written Into the Thesis / 可写入论文的实验点

Recommended main result:

```text
Two-stage ResNet34-U-Net
Stage2 pos_weight = 12
global OOF threshold = 0.80
global OOF min_area = 24
Dice = 0.756876
IoU = 0.633878
defect image recall = 0.995261
normal FPR = 0.000000
```

Recommended ablation table:

| ablation | include? | reason |
| --- | --- | --- |
| `pos_weight` 6/8/12 | yes | Shows the effect of positive-pixel class weighting under severe pixel imbalance. |
| normal FP loss 0.03/0.05 | yes | Useful negative ablation showing that explicit normal penalty did not improve the final objective. |
| deep supervision | yes | Shows an architectural auxiliary-head attempt; improved IoU slightly but worsened balanced metrics. |
| boundary auxiliary loss | yes | Relevant to thin crack boundaries; did not improve final result. |
| PatchDataset cache | method/runtime note | It is an engineering optimization, not a metric-changing model contribution. |
| hard-normal replay cap | method/stability note | Important for stable sampling and fair training. |
| fixed train-time threshold | method/stability note | Important to separate checkpoint selection from post-processing search. |

Current bottleneck:

中文：

U-Net 的主要瓶颈不是 normal false positives，因为最佳 `pos_weight=12` 在 pooled OOF 上 normal FPR 为 0。进一步提升更可能来自数据层面和误差模式层面：更多标注数据、更严格的视频级划分、更精细的边界标注质量检查、以及针对小 phone crack 的专门增强或采样策略。

English:

The main bottleneck of the current U-Net is not normal false positives, because the best `pos_weight=12` model has zero normal FPR in pooled OOF evaluation. Further improvement is more likely to come from data and error-mode improvements: more labeled data, stricter video-level splitting, better boundary annotation quality checks, and targeted augmentation or sampling strategies for very small phone cracks.

Limitations to mention:

| limitation | 中文 | English |
| --- | --- | --- |
| image-level CV | 当前 fold 是 image-level random split，不是 video-level split。 | Current folds are image-level random splits, not video-level splits. |
| broken unlabeled | broken 没有监督 mask，不能纳入 Dice/IoU。 | Broken images have no supervised masks and cannot be included in Dice/IoU. |
| small dataset | 总 labeled crack 只有 263 张。 | Only 263 labeled crack images are available. |
| sparse masks | 裂纹像素占比极低，metric 对小错误敏感。 | Crack masks are extremely sparse, making metrics sensitive to small errors. |
| threshold dependence | 最终 mask 依赖 threshold/min_area 后处理。 | Final masks depend on threshold/min_area post-processing. |

## 26. Glossary / 术语表

| term | 中文 | English |
| --- | --- | --- |
| ROI | 裁剪后的感兴趣区域图像。 | Cropped region-of-interest image. |
| LabelMe JSON | polygon 标注文件。 | Polygon annotation file. |
| binary mask | 0/1 像素级监督图。 | Pixel-level 0/1 supervision map. |
| trainval | 用于交叉验证训练和验证的数据。 | Data used for cross-validation training and validation. |
| holdout | 不参与训练调参，只用于最终推理或测试的数据。 | Data excluded from training/tuning and used for final inference or testing. |
| OOF | out-of-fold，每张图由没训练过它的 fold 模型预测。 | Out-of-fold prediction by a model that did not train on that image. |
| normal FPR | normal 图中出现任意正预测的比例。 | Fraction of normal images with any positive prediction. |
| defect image recall | defect 图中出现任意正预测的比例。 | Fraction of defect images with any positive prediction. |
| hard normal | 被当前模型误检的 normal 图。 | Normal image that produces a false positive under the current model. |
| min_area | 后处理中保留连通域的最小面积。 | Minimum connected-component area kept during post-processing. |
| `pos_weight` | BCE 中正像素权重。 | Positive-pixel weight in BCE. |
| deep supervision | decoder 中间层辅助监督。 | Auxiliary supervision on intermediate decoder outputs. |
| boundary auxiliary loss | 辅助边界预测 loss。 | Auxiliary loss for boundary prediction. |

## 27. Minimal Reproduction Commands / 最小复现实验命令

中文：

以下命令展示完整流程。服务器上运行时请先加载正确 PyTorch module，并在服务器重新生成 manifest。

English:

The commands below show the full pipeline. On the server, load the correct PyTorch module first and regenerate the manifest on the server.

```bash
# 1. Prepare manifest and masks
python scripts/prepare_samples.py --test-ratio 0.20 --test-seed 2026

# 2. Build Stage1 patch indexes
python scripts/build_patch_index.py --config configs/stage1.yaml

# 3. Train one Stage1 fold
python scripts/train_stage1.py --config configs/stage1.yaml --fold 0

# 4. Train one Stage2 fold
python scripts/train_stage2.py --config configs/stage2.yaml --fold 0

# 5. Search pooled OOF post-processing after all folds finish
python scripts/search_oof_postprocess.py --config configs/stage2.yaml --folds 0,1,2,3

# 6. Infer holdout
python scripts/infer_holdout.py --config configs/stage2.yaml --fold 0

# 7. Render validation error analysis
python scripts/visualize_error_analysis.py --config configs/stage2.yaml --fold 0
```

P0 A40 experiment commands:

```bash
bash scripts/run_p0_a40_parallel.sh
bash scripts/run_p0_a40_second_batch.sh
bash scripts/run_p0_a40_normal_fp_loss_ablation.sh
bash scripts/run_p0_a40_third_batch.sh
```
