# 原始两阶段 U-Net 模型数据流说明

> Current entry note: active training now uses only `configs/canonical_baseline.yaml`.
> Older config names in this document describe historical baseline runs and are
> not active project entry points.

本文档只描述原始 baseline 工程，也就是 **ResNet34 encoder + U-Net decoder 的两阶段分割模型**。这里的“原模型”不包含后面新增的 Transformer bottleneck、hard-negative prototype cross-attention、skip attention gate、deep supervision 或 boundary auxiliary 等实验结构。

原模型的核心思想是：

1. 先用 Stage1 在局部 patch 上学习“裂纹长什么样”，解决裂纹像素极少、全图训练时正样本比例太低的问题。
2. 再用 Stage2 把 Stage1 学到的卷积特征迁移到完整 ROI 图像上，学习全局上下文，并通过 normal 样本和 hard-normal replay 控制正常图像上的误检。
3. 最后不直接使用固定 0.5 阈值，而是在 pooled out-of-fold validation 结果上搜索一个全局的 threshold/min_area，再用这个固定设置做 holdout 推理。

对应的主要代码入口是：

| 功能 | 文件 |
| --- | --- |
| 原始数据扫描、LabelMe polygon 转 mask、生成样本 manifest | `scripts/prepare_samples.py` |
| Stage1 patch index 构造 | `scripts/build_patch_index.py` |
| 图像、mask、patch 数据集与 transform | `src/datasets.py` |
| 原始 ResNet34-U-Net 模型 | `src/model.py` |
| BCE + Dice loss | `src/losses.py` |
| Stage1 patch 训练 | `scripts/train_stage1.py` |
| Stage2 full-image 训练 | `scripts/train_stage2.py` |
| hard-normal mining / Stage1 replay mining | `src/mining.py` |
| validation、OOF、post-processing search | `scripts/evaluate_val.py`, `scripts/search_oof_postprocess.py` |
| holdout ensemble 推理 | `scripts/infer_holdout_ensemble.py` |

---

## 1. 总体数据流

整个工程的数据流可以概括为：

```text
Raw ROI image + LabelMe JSON
  -> generated binary mask
  -> manifests/samples.csv
  -> trainval / holdout split
  -> 4-fold cross-validation split
  -> Stage1 patch index
  -> PatchDataset + Stage1 DataLoader
  -> Stage1 ResNet34-U-Net patch training
  -> best_stage1.pt
  -> ROIDataset + Stage2 DataLoader
  -> Stage2 full-image training initialized from Stage1
  -> best_stage2.pt for each fold
  -> per-fold validation export
  -> pooled OOF threshold/min_area search
  -> holdout fold-ensemble inference
```

从一张图片的角度看，数据经历的是：

```text
image.png
label.json 或空 label
  -> binary mask, shape H x W, value in {0,1}
  -> sample row metadata
  -> Dataset.__getitem__()
  -> image tensor [3,H,W], mask tensor [1,H,W]
  -> DataLoader batch [B,3,H,W], [B,1,H,W]
  -> model logits [B,1,H,W]
  -> sigmoid probability map [B,1,H,W]
  -> threshold + connected-component filtering
  -> final binary prediction mask
```

---

## 2. 原始数据与 label 的含义

原始数据集目录为：

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

三类文件夹在工程中的角色不同：

| 文件夹 | 工程角色 |
| --- | --- |
| `crack` | 有监督正样本。每张图片需要同名 LabelMe JSON，JSON 中的 polygon 被转换成二值裂纹 mask。 |
| `normal` | 有监督负样本。没有裂纹 polygon，训练时使用全 0 mask。 |
| `broken` | 当前工程里视为 unlabeled holdout。因为没有像素级 mask，所以不参与 Stage1/Stage2 监督训练和定量 validation。 |

当前默认数据统计见 `DATASET_DESCRIPTION.md`，关键点是：

| split | crack | normal | broken | total | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| `trainval` | 211 | 59 | 0 | 270 | Stage1/Stage2 training and CV validation |
| `holdout` | 52 | 15 | 107 | 174 | final inference / qualitative inspection |

裂纹 mask 非常稀疏。很多 crack 图像中，正像素只占 640x640 ROI 的 1% 以下。这就是为什么工程没有直接只用全图训练，而是设计了 Stage1 patch training。

---

## 3. 从 LabelMe JSON 到 binary mask

入口脚本是 `scripts/prepare_samples.py`。

对于一张 crack 图像：

```text
image_path = dataset0505_crop640_roi/<device>/crack/<name>.png
json_path  = dataset0505_crop640_roi/<device>/crack/<name>.json
mask_path  = generated_masks/<device>/crack/<name>.png
```

LabelMe JSON 中的每个 shape 都包含 polygon points。代码读取所有 polygon，并在一张灰度图上填充 polygon 区域：

```python
mask = Image.new("L", (image_width, image_height), 0)
draw = ImageDraw.Draw(mask)
for shape in data.get("shapes", []):
    polygon = [(float(x), float(y)) for x, y in shape["points"]]
    draw.polygon(polygon, fill=255)
```

数学上，若一张图的 polygon 集合为 $\mathcal{P}=\{P_1,P_2,\ldots,P_K\}$，则生成的监督 mask 是：

$$
Y(x,y)=
\begin{cases}
1, & (x,y)\in \bigcup_{k=1}^{K} P_k \\
0, & \text{otherwise}
\end{cases}
$$

在文件里正像素存成 255，背景为 0；在 Dataset 里再统一二值化为 `{0,1}`。

对于一张 normal 图像：

```text
mask_path = ""
```

它没有 LabelMe JSON，也不生成 mask 文件。后面 `ROIDataset` 或 `PatchDataset` 读取时，如果 `mask_path` 为空，就创建一张和图片同尺寸的全 0 mask。也就是说 normal 样本仍然是有监督训练样本，只是它的监督目标是“整张图没有裂纹”。

对于 broken 图像：

```text
sample_type = broken_unlabeled
split = holdout
mask_path = ""
```

它可以被推理，但不用于训练和定量 Dice/IoU。

---

## 4. 样本 manifest: samples.csv

`prepare_samples.py` 的输出是 `manifests/samples.csv`。这是后面所有训练和评估共同使用的样本清单。

每一行大致表示一张原始 ROI 图像：

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 唯一样本 id，由 device、class、stem 组合生成 |
| `image_name` | 图像文件名 |
| `image_path` | 图像绝对路径 |
| `mask_path` | crack mask 路径；normal/broken 为空 |
| `json_path` | LabelMe JSON 路径；只有 crack 通常存在 |
| `sample_type` | `defect`, `normal`, `broken_unlabeled` |
| `is_labeled` | crack 为 true，normal/broken 为 false |
| `source_split` | 来源说明，例如 `trainval_pool` 或 `broken_folder` |
| `device` | `camera` 或 `phone` |
| `defect_class` | 原始文件夹类别，`crack`, `normal`, `broken` |
| `holdout_reason` | holdout 原因，例如 `test_split` 或 `broken_unlabeled` |
| `cv_fold` | 0,1,2,3 中的一个；只给 trainval 样本分配 |
| `split` | `trainval` 或 `holdout` |

这个 manifest 的作用很重要：工程后续不再直接“猜”数据集结构，而是通过 `samples.csv` 明确知道每张图的输入、label、split 和 fold。

---

## 5. Train/Holdout 和 4-fold 交叉验证

`prepare_samples.py` 先从 crack/normal 中按默认比例保留 20% 作为 holdout：

```text
--test-ratio 0.20
--test-seed 2026
```

然后对剩下的 trainval 样本分配 4 个 CV fold：

```text
--n-folds 4
--seed 42
```

fold 分配按 `(device, sample_type, defect_class)` 分组后随机打散，再轮流分配到 0/1/2/3。这样可以保持 camera/phone、crack/normal 的大致平衡。

在第 k fold 中：

```text
validation = cv_fold == k
training   = cv_fold != k
```


对应的超参数是 `test_ratio`、`test_seed`、`n_folds` 和 `seed`。设所有 labeled crack 图像为 $D_c$，normal 图像为 $D_n$，broken unlabeled 图像为 $D_b$。`prepare_samples.py` 先在每个分组 $g=(device, sample\_type, defect\_class)$ 内抽 holdout：

$$
H_g = \operatorname{Sample}_{test\_seed}\left(D_g,\; \operatorname{round}(test\_ratio \cdot |D_g|)\right)
$$

$$
T_g = D_g \setminus H_g
$$

其中 $H_g$ 是 holdout，$T_g$ 是 trainval pool。`broken` 没有像素级 label，直接进入 holdout：

$$
D_b \rightarrow H
$$

然后 `n_folds=4` 把 trainval 分成 4 折。代码同样按分组 shuffle，再轮流分配：

$$
cv\_fold(x_i) = i \bmod K, \quad K = n\_folds
$$

因此第 $k$ 折的训练/验证集合是：

$$
V_k = \{x \in T \mid cv\_fold(x)=k\}
$$

$$
R_k = \{x \in T \mid cv\_fold(x)\neq k\}
$$

这几个参数会改变样本划分本身。如果改了 `test_ratio`、`test_seed`、`n_folds` 或 fold seed，后面的 Stage1、Stage2、OOF 都应该整体重跑，不能直接和旧划分上的结果公平比较。

`src/samples.py` 中的 `split_samples_for_fold()` 返回：

```text
defect_train_rows
defect_val_rows
normal_train_rows
normal_val_rows
```

这四组数据会分别进入 Stage1 patch 构造、Stage2 full-image 训练和 validation。

---

## 6. Stage1 patch index 构造

Stage1 的目标不是直接预测整张图，而是让模型大量看到“局部裂纹”和“局部易混淆背景”。入口脚本是：

```text
scripts/build_patch_index.py
```

输入：

```text
manifests/samples.csv
configs/stage1_p0_a40.yaml
```

输出：

```text
manifests/stage1_fold0_train_index.csv
manifests/stage1_fold0_val_index.csv
...
manifests/stage1_fold3_train_index.csv
manifests/stage1_fold3_val_index.csv
manifests/stage1_patch_summary.json
```


这一阶段的核心超参数是 crop size、每类 patch 数量、near-miss/hard-negative 的空间约束，以及 patch 去重阈值。对一张 crack 图，令 mask 为 $Y\in\{0,1\}^{H\times W}$，先找连通域：

$$
C = \operatorname{ConnectedComponents}(Y)
$$

然后按面积排序，只保留前 `max_components_per_image` 个：

$$
C' = \operatorname{TopM}(C), \quad M=max\_components\_per\_image
$$

一个 patch crop window 记为 $w=(x,y,s)$，其中 $(x,y)$ 是左上角，$s$ 是 `crop_size`。窗口中的 mask 是：

$$
Y_w = Y[y:y+s,\;x:x+s]
$$

positive patch 的必要条件是：

$$
\sum Y_w > 0
$$

negative patch 的必要条件是：

$$
\sum Y_w = 0
$$

这里 `crop_size` 控制原图上看到的上下文范围，`patch_out_size` 控制输入模型前 resize 后的分辨率。也就是说，`crop_size=448, patch_out_size=384` 会让模型在同样输入尺寸下看到更大的上下文，但细节被压缩得更多。

Stage1 index 的每一行不再是一张原图，而是一个 patch 裁剪窗口：

| 字段 | 含义 |
| --- | --- |
| `patch_id` | patch 唯一 id |
| `base_sample_id` | 来源原图 id |
| `image_path` | 来源原图路径 |
| `mask_path` | 来源 mask 路径，normal 为空 |
| `patch_type` | 具体 patch 类型 |
| `patch_family` | 归并后的 patch family |
| `crop_x`, `crop_y` | 左上角坐标 |
| `crop_size` | 原图上裁剪窗口大小 |
| `out_size` | 输入模型前 resize 后大小，原始配置为 384 |
| `component_id` | 来源裂纹连通域 id |
| `component_area_px` | 来源裂纹连通域面积 |

### 6.1 crack 图像上的 positive patches

对于 crack 图像，先从 mask 中找连通域：

```text
mask -> connected components -> component bbox / center / area
```

然后围绕裂纹 component 生成多种正样本 patch：

| patch_type | 目的 |
| --- | --- |
| `positive_center` | 把裂纹放在 patch 中心，帮助模型学习典型裂纹外观。 |
| `positive_shift` | 随机移动裁剪框，但保证包含裂纹，增强位置不变性。 |
| `positive_context` | 使用较大 crop，让模型看到裂纹周围上下文。 |
| `positive_boundary` | 故意让裂纹靠近 patch 边缘，模拟真实全图中裂纹被裁剪或局部出现的情况。 |

配置中的典型 crop size 是：

```text
positive_center_crop_sizes:   [320, 384, 448]
positive_shift_crop_sizes:    [320, 384, 448]
positive_context_crop_sizes:  [384, 448]
positive_boundary_crop_sizes: [320, 384, 448]
```

不管原始 crop size 是 320、384 还是 448，进入 Stage1 模型前都会 resize 到：

```text
patch_out_size = 384
```


这些 patch 由下面几组超参数控制：

| patch type | crop size 参数 | count 参数 | 影响 |
| --- | --- | --- | --- |
| `positive_center` | `positive_center_crop_sizes` | `positive_center_count_per_image` | 裂纹更靠近 patch 中心 |
| `positive_shift` | `positive_shift_crop_sizes` | `positive_shift_count_per_image` | 裂纹位置随机偏移，增强位置鲁棒性 |
| `positive_context` | `positive_context_crop_sizes` | `positive_context_count_per_image` | 增加上下文范围 |
| `positive_boundary` | `positive_boundary_crop_sizes` | `positive_boundary_count_per_image` | 模拟裂纹靠近裁剪边界的情况 |

正样本 patch 数量还会被 `max_positive_patches_per_image` 截断：

$$
N_{pos,image} \leq max\_positive\_patches\_per\_image
$$

### 6.2 crack 图像上的 defect-negative patches

同一张 crack 图像里也会采样不含裂纹的负 patch。它们比普通 normal 更难，因为背景来自有缺陷图像附近。

| patch_type | 目的 |
| --- | --- |
| `near_miss_negative` | 在裂纹 bbox 周围但不覆盖正像素，学习“接近裂纹但不是裂纹”的背景。 |
| `hard_negative` | 从 crack 图像中远离裂纹 bbox 的区域随机采样，学习缺陷图像背景。 |

代码会检查裁剪窗口中是否存在正像素。如果窗口中有正像素，就不会作为 negative patch。


near-miss negative 和 hard negative 的差别在于空间约束。设某个裂纹连通域的 bbox 为 $B_c$。near-miss negative 要靠近它，但不能覆盖正像素：

$$
\sum Y_w = 0
$$

$$
window(w) \cap \operatorname{expand}(B_c, near\_miss\_margin\_max) \neq \varnothing
$$

其中距离范围由 `near_miss_margin_min` 和 `near_miss_margin_max` 控制。hard negative 使用整张 crack mask 的全局 bbox $B_Y$，并避开扩张后的 forbidden region：

$$
B_{forbid}=\operatorname{expand}(B_Y, hard\_negative\_safety\_margin)
$$

$$
\sum Y_w = 0, \quad window(w) \cap B_{forbid}=\varnothing
$$

因此 `hard_negative_safety_margin` 越大，hard negative 越远离真实裂纹；越小，则更可能采到靠近裂纹的困难背景。

### 6.3 normal 图像上的 normal-negative patches

从 normal 图像随机采样：

```text
patch_type = normal_negative
patch_family = normal_negative
mask = all-zero
```

normal-negative patch 的数量被设置成接近 defect-negative patch 的数量，用来平衡模型对正常背景的认识。


代码里这个数量是直接由 defect-negative 数量决定的：

$$
N_{normal\_negative}=N_{near\_miss\_negative}+N_{hard\_negative}
$$

所以 normal negative 不是一个完全独立的数量旋钮。若你增加 `near_miss_negative_count_per_image` 或 `hard_negative_count_per_image`，normal-negative 也会随之增加，以保持 crack 图背景和 normal 图背景的大体平衡。

### 6.4 patch 去重

同一张原图上生成多个 patch 时，会计算裁剪窗口之间的 IoU。如果新窗口和已有窗口的 IoU 超过：

```text
patch_dedup_iou = 0.70
```

就拒绝这个新 patch，避免训练集中出现大量几乎一样的 crop。


两个 crop window 的 IoU 为：

$$
IoU(w_a,w_b)=\frac{|w_a\cap w_b|}{|w_a\cup w_b|}
$$

当新窗口满足：

$$
IoU(w_{new},w_{old}) > patch\_dedup\_iou
$$

它会被拒绝。`patch_dedup_iou` 越小，patch 越多样，但也更可能采不到足够数量；越大，允许更多相似 patch。

---

## 7. Stage1 PatchDataset 数据流

Stage1 使用 `src/datasets.py` 里的 `PatchDataset`。

对于一个 patch row，`PatchDataset.__getitem__()` 做：

```text
1. read image_path as RGB numpy array
2. if mask_path exists:
       read mask_path as binary mask
   else:
       create empty mask with same H,W
3. read crop_x, crop_y, crop_size, out_size
4. crop image[y:y+crop_size, x:x+crop_size]
5. crop mask[y:y+crop_size, x:x+crop_size]
6. apply Stage1 transform
7. return image tensor, mask tensor, and patch metadata
```

输出字典主要包括：

```text
image: [3, 384, 384] float tensor
mask:  [1, 384, 384] float tensor
patch_id
base_sample_id
patch_type
patch_family
sample_type
component_id
component_area_px
is_replay
```

Stage1 的 image/mask transform 是同步的：

| 操作 | 图像 | mask |
| --- | --- | --- |
| resize | bilinear | nearest |
| horizontal flip | yes | yes |
| vertical flip | yes | yes |
| rotate | bilinear | nearest |
| brightness/contrast/gamma | yes | no |
| blur/noise | yes | no |
| tensor conversion | yes | yes |
| ImageNet normalize | yes | no |

这里 mask 始终保持二值，不会因为插值变成连续 label。

---

## 8. Stage1 DataLoader 设计

入口在 `scripts/train_stage1.py` 的 `build_stage1_loaders_from_rows()`。

Stage1 train loader 的 batch 是 patch batch：

```text
image batch: [B, 3, 384, 384]
mask batch:  [B, 1, 384, 384]
```

原始配置：

```text
batch_size = 128
num_workers = 8
patch_out_size = 384
```

Stage1 默认使用 `WeightedRandomSampler` 做 family-level balancing。配置里的数值是不同 patch family 的**相对采样权重**：

```text
positive:        0.50
defect_negative: 0.25
normal_negative: 0.25
replay:          0.15 if replay exists
```

注意这里的比例不是简单复制数据，也不是严格保证每个 batch 精确满足 50/25/25。代码会给每个 family 内的样本分配权重，然后用 `WeightedRandomSampler(replacement=True)` 采样 `len(train_rows)` 个 patch。没有 replay 时，期望采样比例大约接近 positive/defect-negative/normal-negative = 0.50/0.25/0.25；有 replay 后，`0.15` 会作为额外相对权重参与归一化，所以实际期望比例大约是 `0.50:0.25:0.25:0.15` 归一化后的结果。这样一个 epoch 中更容易稳定看到足够多的 positive patch，而不是被 negative 背景淹没。


形式化地说，设第 $i$ 个 patch 的 family 为 $f_i$，某个 family 的样本数为 $n_f$，配置中的目标相对权重为 $\alpha_f$。代码给每个 patch 的 sampler weight 是：

$$
w_i=\frac{\alpha_{f_i}}{n_{f_i}}
$$

归一化后，第 $i$ 个 patch 被采样的概率是：

$$
p_i=\frac{w_i}{\sum_j w_j}
$$

因此某个 family 的期望采样比例是：

$$
P(family=f)=\frac{\alpha_f}{\sum_g \alpha_g}
$$

没有 replay 时，$\alpha=(0.50,0.25,0.25)$，期望就是 positive / defect-negative / normal-negative = 50% / 25% / 25%。有 replay 后，`stage1_replay_ratio=0.15` 会作为额外相对权重参与归一化。

Stage1 还会在每个 epoch 用不同的 seed 重建 loader，减少重复增强模式：

$$
seed_e = seed_{base} + 1009e
$$

如果启用了 patch worker cache，DataLoader worker 会在自己的进程里缓存已经读过的 image/mask，减少重复 I/O。这只改变速度，不改变训练样本定义。

---

## 9. 原始 ResNet34-U-Net 模型结构

模型定义在 `src/model.py`。

原始 baseline 是：

```text
UNetResNet34(
    encoder = torchvision ResNet34,
    decoder = bilinear upsampling + skip connection + ConvBlock,
    output = 1-channel binary segmentation logit
)
```

对于 Stage2 的 640x640 输入，主要 feature map 尺寸是：

| 名称 | 来源 | 通道数 | 空间尺寸 |
| --- | --- | ---: | --- |
| input | RGB image | 3 | 640 x 640 |
| x0 | ResNet conv1 + BN + ReLU | 64 | 320 x 320 |
| x1 | maxpool + layer1 | 64 | 160 x 160 |
| x2 | layer2 | 128 | 80 x 80 |
| x3 | layer3 | 256 | 40 x 40 |
| x4 | layer4 | 512 | 20 x 20 |
| center | two 3x3 conv block | 512 | 20 x 20 |
| d4 | upsample center + concat x3 | 256 | 40 x 40 |
| d3 | upsample d4 + concat x2 | 128 | 80 x 80 |
| d2 | upsample d3 + concat x1 | 64 | 160 x 160 |
| d1 | upsample d2 + concat x0 | 64 | 320 x 320 |
| logits | upsample d1 to input + seg head | 1 | 640 x 640 |

Decoder block 的形式是：

```text
upsample previous feature to skip size
concat([upsampled feature, encoder skip], channel dimension)
3x3 conv + BN + ReLU
3x3 conv + BN + ReLU
```

最终 segmentation head 是：

```text
Conv2d(64 -> 32, kernel=3, padding=1)
BatchNorm2d(32)
ReLU
Conv2d(32 -> 1, kernel=1)
```

模型输出不是概率，而是 logit：

$$
Z=f_{\theta}(X), \quad Z\in\mathbb{R}^{B\times1\times H\times W}
$$

$$
P=\sigma(Z)
$$

原始 baseline 中以下模块全部关闭：

| 模块 | 原模型是否使用 |
| --- | --- |
| Transformer bottleneck | no |
| Prototype cross-attention | no |
| Skip attention gate | no |
| Deep supervision | no |
| Boundary auxiliary head | no |
| Normal FP auxiliary loss | no |

这些模块虽然在当前代码里已经存在，是后续消融/改进实验用的，不属于这里的原始模型。

---

## 10. Stage1 训练流程

Stage1 入口是：

```text
python scripts/train_stage1.py --config configs/stage1_p0_a40.yaml --fold <fold>
```

每个 fold 独立训练一个 Stage1 模型：

```text
outputs/experiments/p0_a40_20260508/stage1/fold0/best_stage1.pt
outputs/experiments/p0_a40_20260508/stage1/fold1/best_stage1.pt
outputs/experiments/p0_a40_20260508/stage1/fold2/best_stage1.pt
outputs/experiments/p0_a40_20260508/stage1/fold3/best_stage1.pt
```

### 10.1 初始化

Stage1 直接构建原始模型：

```python
model = build_model(pretrained=True)
```

也就是说 encoder 尝试加载 torchvision 的 ImageNet pretrained ResNet34 权重。decoder 是新初始化。

### 10.2 encoder freeze

配置中：

```text
freeze_encoder_epochs = 3
```

前 3 个 epoch 冻结 encoder，只训练 center、decoder 和 segmentation head。这样可以避免一开始 patch 数据的强梯度立刻破坏 ImageNet backbone 的通用低层特征。

第 4 个 epoch 起解冻 encoder，整个网络一起 fine-tune。


写成公式，前 `freeze_encoder_epochs` 个 epoch 中：

$$
requires\_grad(\theta_{encoder}) = false
$$

$$
requires\_grad(\theta_{decoder}) = true
$$

同时 encoder 的 BatchNorm 进入 eval mode，防止 patch 分布更新 running mean/variance。这个细节很重要，因为如果只冻结参数但继续更新 BatchNorm running stats，encoder 仍然会被 patch 数据分布悄悄改变。

### 10.3 loss

Stage1 使用 BCE + Dice。BCE 负责逐像素分类稳定性，Dice 负责区域重叠。令 logit 为 $z$，概率为 $p=\sigma(z)$，标签为 $y\in\{0,1\}$。带 `pos_weight` 的 BCE 是：

$$
\mathcal{L}_{BCE}(z,y)=-pos\_weight\cdot y\log(p)-(1-y)\log(1-p)
$$

Dice 系数是：

$$
Dice(P,Y)=\frac{2\sum(P\cdot Y)+\epsilon}{\sum P+\sum Y+\epsilon}
$$

Dice loss 是：

$$
\mathcal{L}_{Dice}=1-Dice(P,Y)
$$

最终 Stage1 segmentation loss 是：

$$
\mathcal{L}_{seg}=bce\_weight\cdot\mathcal{L}_{BCE}+dice\_weight\cdot\mathcal{L}_{Dice}
$$

当前配置为 `bce_weight=0.5`、`dice_weight=0.5`、`pos_weight=12`，所以：

$$
\mathcal{L}_{seg}=0.5\mathcal{L}_{BCE}+0.5\mathcal{L}_{Dice}
$$

`pos_weight` 越大，模型越重视正像素召回；但过大也可能提高 normal 或 defect-negative patch 上的误检。

### 10.4 optimizer and scheduler

训练使用 AdamW，并且 encoder/decoder 使用不同学习率：

```text
encoder_lr = 5e-5
decoder_lr = 2e-4
weight_decay = 1e-4
```

因为 decoder 是新训练的，学习率更高；encoder 是 pretrained backbone，学习率更低。

scheduler 使用 ReduceLROnPlateau，在 validation score 停滞时降低学习率。


参数被分成 encoder 和 decoder 两组：

$$
\theta=\theta_{encoder}\cup\theta_{decoder}
$$

两组学习率分别是：

$$
lr(\theta_{encoder})=encoder\_lr
$$

$$
lr(\theta_{decoder})=decoder\_lr
$$

当 validation score 连续 `lr_patience` 次没有提升时：

$$
lr_{new}=\max(lr_{old}\cdot lr\_factor,\;min\_lr)
$$

### 10.5 Stage1 validation

Stage1 validation 仍在 patch 上做。主要指标包括：

| 指标 | 含义 |
| --- | --- |
| `patch_dice_all` | 所有 patch 的平均 Dice |
| `patch_dice_pos_only` | 只在 positive patch 上算 Dice |
| `positive_patch_recall` | positive patch 中是否预测出至少一个正像素 |
| `negative_patch_fpr` | negative patch 中出现正预测的比例 |

Stage1 checkpoint 选择使用一个带 negative false-positive penalty 的分数。默认：

```text
stage1_target_negative_fpr = 0.20
stage1_negative_fpr_penalty = 0.5
```

也就是说 Stage1 首先要学会找裂纹，但不能在 negative patch 上误检太多。公式是：

$$
S_{stage1}=Dice_{pos}-\lambda_{neg}\max(0,FPR_{neg}-FPR_{target})
$$

其中 $\lambda_{neg}$ 对应 `stage1_negative_fpr_penalty`，$FPR_{target}$ 对应 `stage1_target_negative_fpr`。

### 10.6 Stage1 replay

Stage1 还有一个 replay 机制。warmup 后，每隔若干 epoch 用当前模型扫描基础 patch：

```text
stage1_replay_warmup_epochs = 3
stage1_replay_refresh_every = 3
stage1_replay_ratio = 0.15
```

它会挑出：

| replay 类型 | 来源 |
| --- | --- |
| `replay_positive_fn` | positive patch 里模型完全漏检的样本 |
| `replay_positive_hard` | positive patch 里 Dice 较差的样本 |
| `replay_defect_negative_fp` | defect-negative patch 上误检的样本 |
| `replay_normal_negative_fp` | normal-negative patch 上误检的样本 |

这些 replay rows 会加入后续 epoch 的训练集，迫使模型反复学习当前最容易错的 patch。


replay 刷新由 `stage1_replay_warmup_epochs` 和 `stage1_replay_refresh_every` 控制。若 epoch 为 $e$，warmup 为 $w$，刷新间隔为 $r$，则刷新条件是：

$$
e\geq w \quad \text{and} \quad (e-w)\bmod r=0
$$

replay 数量由 `stage1_replay_ratio` 和 `stage1_max_replay_ratio` 控制：

$$
N_{replay}=\min\left(\operatorname{round}(stage1\_replay\_ratio\cdot N_{base}),\;\operatorname{round}(stage1\_max\_replay\_ratio\cdot N_{base})\right)
$$

positive replay 的难度主要来自低 Dice 或完全漏检：

$$
score_{pos}=1-Dice(\hat{Y},Y)+\mathbf{1}[\sum\hat{Y}=0 \land \sum Y>0]
$$

negative replay 的难度主要来自误检比例和最大概率：

$$
score_{neg}=\frac{\sum\hat{Y}}{|\hat{Y}|}+\max(P)
$$

---

## 11. Stage2 full-image 数据流

Stage2 的输入不再是 patch index，而是 `samples.csv` 中的完整 ROI 图像。

入口：

```text
python scripts/train_stage2.py --config configs/stage2_p0_a40_e50_bs48_pw12.yaml --fold <fold>
```

数据读取使用 `ROIDataset`：

```text
1. read full image as RGB
2. if mask_path exists:
       read binary mask
   else:
       create all-zero mask
3. resize image and mask to image_size
4. apply training/eval transform
5. return image tensor, mask tensor, sample metadata
```

原始 Stage2 配置：

```text
image_size = 640
batch_size = 48
epochs = 50
num_workers = 8
```

Stage2 batch：

```text
image batch: [B, 3, 640, 640]
mask batch:  [B, 1, 640, 640]
```

### 11.1 从 Stage1 checkpoint 初始化

Stage2 不是从 ImageNet 重新开始，而是从同一个 fold 的 Stage1 checkpoint 初始化：

```text
stage1_checkpoint_template:
outputs/experiments/p0_a40_20260508/stage1/fold{fold}/best_stage1.pt
```

代码中：

```python
model = build_model_from_config(cfg)
load_checkpoint(stage1_checkpoint_path, model, strict=True)
```

虽然 Stage1 输入是 384x384 patch，Stage2 输入是 640x640 full ROI，但这个迁移是成立的，因为 ResNet34-U-Net 是 fully convolutional。卷积权重不依赖固定输入尺寸，只要求通道结构一致。

### 11.2 每个 epoch 的训练样本组成

Stage2 每个 epoch 都重新构造 train rows：

```text
epoch_train_rows =
    all defect_train_rows
  + sampled normal rows
  + sampled hard-normal rows, if available
```

其中：

```text
random_normal_k_factor = 1.0
```

表示 normal budget 大约等于 defect train 图像数量。这样 Stage2 不是只看 crack 图像，也会稳定看到 normal 图像。

如果 hard-normal pool 已经存在，则 normal budget 会拆成两部分：

```text
normal budget = random normals + hard normals
```

原始配置：

```text
use_hard_normal_replay = true
stage2_hard_normal_ratio = 0.40
hard_normal_max_repeats_per_epoch = 2
hard_normal_warmup_epochs = 2
hard_normal_refresh_every = 2
hard_normal_pool_factor = 3.0
```

也就是说，warmup 后模型会定期找正常图像中最容易误检的样本，并在后续 epoch 中更频繁地训练它们。但每张 hard-normal 图最多重复 2 次，避免一个很小的 hard pool 被过度重复。


对应公式如下。设当前 fold 的 defect training image 数为 $N_d$，normal budget 为：

$$
B_n=\operatorname{round}(random\_normal\_k\_factor\cdot N_d)
$$

如果 hard-normal pool 非空，期望 hard-normal 数量为：

$$
R_h=\operatorname{round}(stage2\_hard\_normal\_ratio\cdot B_n)
$$

但实际采样会被 hard pool size 和重复上限截断：

$$
H_{target}=\min(R_h,\;|H_{pool}|\cdot hard\_normal\_max\_repeats\_per\_epoch)
$$

随机 normal 数量为：

$$
R_{random}=B_n-H_{target}
$$

所以每个 epoch 的训练集合是：

$$
E=R_{defect}\cup R_{random\_normal}\cup R_{hard\_normal}
$$

---

## 12. Stage2 hard-normal mining

Hard-normal mining 在 `src/mining.py` 的 `build_hard_normal_pool()` 中实现。

流程是：

```text
1. 用当前 Stage2 模型扫描 normal_train_rows
2. 对每张 normal 图得到 prob_map
3. 使用当前 validation threshold/min_area 得到 pred_mask
4. 如果 pred_mask 中有正像素，则这是一个 false-positive normal
5. 计算该 normal 图的 hard score
6. 按 hard score 排序，选出 hard-normal pool
7. 写入 save_dir/hard_normal/
```

hard score 由几个 false-positive 强度组成：

$$
score_i = 10^6\cdot A^{largest}_{fp,i}+N_{fp,i}+\max(P_i)
$$

其中 $A^{largest}_{fp,i}$ 是最大 false-positive 连通域面积，$N_{fp,i}$ 是 false-positive 像素总数，$\max(P_i)$ 是这张 normal 图上的最大预测概率。

所以排序优先级是：

1. 最大误检连通域面积越大，越 hard。
2. 总误检像素越多，越 hard。
3. 最大概率越高，越 hard。

输出文件包括：

```text
hard_normal/hard_normal_latest.csv
hard_normal/hard_normal_epochXXX.csv
hard_normal/hard_normal_latest_summary.json
```

这个机制的论文解释是：normal 图像上的误检往往来自局部纹理、边缘、光照或接触线结构。hard-normal replay 是一种在线困难负样本挖掘，它让模型在后续训练中重复看到当前最像裂纹的正常区域，从而降低部署时的 false positive。

---

## 13. Stage2 loss、checkpoint selection 和 validation

Stage2 原始 baseline 仍使用 BCE + Dice，形式和 Stage1 相同：

$$
\mathcal{L}_{seg}=0.5\mathcal{L}_{BCE}+0.5\mathcal{L}_{Dice}
$$

原始模型里：

```text
normal_fp_loss_weight = 0.0
deep_supervision_weight = 0.0
boundary_aux_weight = 0.0
```

所以没有额外 normal FP loss、没有深监督、没有边界辅助分支。


如果启用 `normal_fp_loss_weight`，它只作用在 target 全 0 的 normal 图上。设 normal 图的预测概率集合为 $P_i$，取 top-k：

$$
k=\operatorname{round}(normal\_fp\_topk\_ratio\cdot |P_i|)
$$

$$
\mathcal{L}_{normal\_fp}=\operatorname{mean}(\operatorname{topk}(P_i,k))
$$

总 loss 变为：

$$
\mathcal{L}=\mathcal{L}_{seg}+normal\_fp\_loss\_weight\cdot\mathcal{L}_{normal\_fp}
$$

当前 baseline 中 `normal_fp_loss_weight=0.0`，所以这一项关闭。

Stage2 训练时的 validation 使用固定：

```text
train_eval_threshold = 0.50
train_eval_min_area = 0
```

这样做的原因是：训练过程中只需要一个稳定的标准选择 checkpoint，不希望每个 epoch 都做很慢的 threshold/min_area search，也不希望 checkpoint selection 被过度调参影响。

Stage2 validation 的主要指标：

| 指标 | 含义 |
| --- | --- |
| `defect_dice` | crack/defect validation 图像上的平均 Dice |
| `defect_iou` | crack/defect validation 图像上的平均 IoU |
| `defect_image_recall` | defect 图像是否至少预测出一个正区域 |
| `normal_fpr` | normal validation 图像中出现任何正预测的比例 |
| `normal_fp_count` | 有 false positive 的 normal 图像数量 |
| `pixel_precision_labeled_micro` | 所有有 label 图像上的 pixel-level micro precision |
| `pixel_recall_labeled_micro` | 所有有 label 图像上的 pixel-level micro recall |
| `component_recall_3px` | 3px tolerance 下的连通域召回 |
| `boundary_f1_3px` | 3px tolerance 下的边界 F1 |

Stage2 的综合 score 是带 normal false-positive 约束的 defect Dice。默认：

```text
target_normal_fpr = 0.10
lambda_fpr_penalty = 2.0
```

$$
S_{stage2}=Dice_{defect}-\lambda_{fpr}\max(0,FPR_{normal}-FPR_{target})
$$

其中 $\lambda_{fpr}$ 对应 `lambda_fpr_penalty`，$FPR_{target}$ 对应 `target_normal_fpr`。

checkpoint selection 的逻辑是：

1. 如果一个结果满足 normal_fpr <= target_normal_fpr，而另一个不满足，优先选满足的。
2. 如果都满足，优先选 defect_dice 高的。
3. Dice 一样时，选 normal_fpr 更低的。
4. 再相同，选 defect_image_recall 更高的。

每个 fold 的 Stage2 输出包括：

```text
best_stage2.pt
last_stage2.pt
history.csv
val_metrics.json
val_per_image.csv
val_postprocess_search.csv
hard_normal/
```

---

## 14. OOF global post-processing search

训练完 4 个 fold 后，每个 trainval 图像都可以被“没有训练过它”的 fold 模型预测一次。这些预测叫 out-of-fold predictions。

入口：

```text
python scripts/search_oof_postprocess.py \
  --config configs/stage2_p0_a40_e50_bs48_pw12.yaml \
  --folds 0,1,2,3
```

流程：

```text
for fold in [0,1,2,3]:
    load fold best_stage2.pt
    predict this fold's validation images from best_stage2.pt
    collect prob_maps, gt_masks, sample_types, image_names

pool all fold validation predictions
for threshold in threshold_grid:
    for min_area in min_area_grid:
        prob_map -> binary mask
        remove components smaller than min_area
        compute metrics
select best global threshold/min_area
```

原始搜索范围：

```text
threshold_grid_start = 0.10
threshold_grid_end   = 0.95
threshold_grid_step  = 0.02
min_area_grid        = [0, 8, 16, 24, 32, 48]
```

这一步会重新加载每个 fold 的 `best_stage2.pt` 并重新预测该 fold 的 validation 图像，而不是直接读取之前 `val_metrics.json` 里的汇总数值。它比较慢的原因是：除了模型 forward 外，还要对所有 OOF probability map 在多个 threshold/min_area 组合上重复做二值化、连通域过滤和指标计算。尤其是 connected-component filtering 是 CPU/Numpy/Scipy 后处理，不是典型 GPU 加速任务。


后处理对每个 probability map $P$ 的形式是：

$$
\hat{Y}_{raw}=\mathbf{1}[P\geq \tau]
$$

$$
\hat{Y}=\operatorname{RemoveSmallComponents}(\hat{Y}_{raw}, a)
$$

其中 $\tau$ 是 threshold，$a$ 是 `min_area`。OOF search 实际上是在候选集合上找：

$$
(\tau^*,a^*)=\arg\max_{\tau\in\mathcal{T},a\in\mathcal{A}} S_{stage2}(\tau,a)
$$

当前最佳配置里：

$$
\mathcal{T}=\{0.10,0.12,\ldots,0.94,0.95\},\quad \mathcal{A}=\{0,8,16,24,32,48\}
$$

这里得到的 $(\tau^*,a^*)$ 是最终推理固定使用的决策参数，不会再根据 holdout 单独调整。

输出：

```text
oof_global_postprocess.json
oof_global_postprocess_search.csv
oof_per_image.csv
```

`oof_global_postprocess.json` 中的 threshold/min_area 是最终推理要使用的全局后处理参数。

---

## 15. Inference 和 holdout ensemble

最终 holdout 推理入口：

```text
python scripts/infer_holdout_ensemble.py \
  --config configs/stage2_p0_a40_e50_bs48_pw12.yaml \
  --folds 0,1,2,3
```

流程：

```text
1. load holdout rows from samples.csv
2. load global OOF threshold/min_area
3. for each fold:
       load fold best_stage2.pt
       predict probability maps for all holdout images
4. average probability maps across folds
5. threshold averaged probability map
6. remove connected components smaller than min_area
7. save probability maps, raw masks, post-processed masks, summary CSV
```

fold ensemble 的概率为：

```text
P_ensemble = (P_fold0 + P_fold1 + P_fold2 + P_fold3) / 4
```

输出目录默认在 `global_postprocess_path` 同级的：

```text
holdout_ensemble/
|- prob_maps/
|- raw_binary_masks/
|- masks/
|- inference_summary.csv
|- holdout_metrics.json, if quantitative holdout masks exist
`- holdout_per_image.csv, if quantitative holdout masks exist
```

其中：

| 输出 | 含义 |
| --- | --- |
| `prob_maps/` | sigmoid 概率图，灰度 PNG |
| `raw_binary_masks/` | threshold 后但未做 min_area 过滤的 mask |
| `masks/` | threshold + min_area 后的最终 mask |
| `inference_summary.csv` | 每张 holdout 图的 max_prob、正像素数量等摘要 |

对 holdout 中的 crack/normal test_split 图像，如果有 mask，可以计算定量指标。对 broken_unlabeled 图像，因为没有 mask，只做 qualitative inference，不进入 Dice/IoU。

---

## 16. 一张 crack 图从输入到输出的完整例子

假设输入是一张 crack 图：

```text
camera/crack/example.png
camera/crack/example.json
```

### 16.1 准备阶段

`prepare_samples.py`：

```text
example.json polygon
  -> generated_masks/camera/crack/example.png
  -> samples.csv row:
       sample_type = defect
       is_labeled = true
       mask_path = generated mask path
       split = trainval or holdout
       cv_fold = 0/1/2/3 if trainval
```

### 16.2 Stage1

`build_patch_index.py`：

```text
read generated mask
find connected components
generate positive_center / shift / context / boundary patches
generate near_miss_negative / hard_negative patches
write crop_x, crop_y, crop_size, out_size to stage1 index CSV
```

`PatchDataset`：

```text
read image and mask
crop patch
resize to 384x384
augment
normalize
return [3,384,384], [1,384,384]
```

`UNetResNet34`：

```text
[B,3,384,384] -> logits [B,1,384,384]
```

loss：

$$
\mathcal{L}_{seg}=0.5\mathcal{L}_{BCE}+0.5\mathcal{L}_{Dice}
$$

输出：

```text
best_stage1.pt
```

### 16.3 Stage2

同一张图如果在当前 fold 的 training set：

```text
ROIDataset reads full 640x640 image and full 640x640 mask
transform
DataLoader batches it with defect and normal images
model initialized from best_stage1.pt
full-image logits [B,1,640,640]
loss against full mask
```

如果它在当前 fold 的 validation set：

```text
model predicts prob_map
prob_map thresholded at train_eval_threshold during training validation
after training, it is also used in fold validation export and OOF search
```

### 16.4 OOF/inference

当这张图属于 fold k 的 validation split：

```text
only fold k model predicts it for OOF
its prob_map joins global OOF search
```

最终通过 OOF search 确定全局后处理参数：

$$
(\tau^*,a^*)=(threshold^*,min\_area^*)
$$

推理时：

```text
logits -> sigmoid prob_map
prob_map >= threshold -> binary mask
remove connected components with area < min_area
save final mask
```

---

## 17. 一张 normal 图从输入到输出的完整例子

假设输入是一张 normal 图：

```text
phone/normal/example.png
```

它没有 LabelMe JSON。

### 17.1 准备阶段

`prepare_samples.py` 写入：

```text
sample_type = normal
is_labeled = false
mask_path = ""
split = trainval or holdout
```

### 17.2 Stage1

`build_patch_index.py` 会从 normal 图随机裁剪：

```text
patch_type = normal_negative
patch_family = normal_negative
mask = all-zero
```

`PatchDataset` 看到 `mask_path=""` 后创建全 0 mask。

Stage1 学到：这些局部纹理不应该被预测成裂纹。

### 17.3 Stage2

Stage2 中 normal 图作为 full-image negative sample。`ROIDataset` 创建全 0 mask：

```text
image: [3,640,640]
mask:  [1,640,640], all zeros
```

训练 loss 会惩罚它上面的正预测。validation 时，如果 final binary mask 有任何正像素，就算这个 normal 图发生了 false positive：

```text
normal_fpr = normal images with pred_has_positive / total normal images
```

### 17.4 hard-normal replay

如果某张 normal 图在当前模型下产生了 false positive，它可能进入 hard-normal pool。之后它会更频繁地出现在训练 batch 中。

这个机制让模型主动修正自己最容易误判的 normal 图。

---

## 18. 原模型中最关键的设计动机

### 18.1 为什么需要两阶段

裂纹像素极少。如果只用 full-image 训练，一个 batch 中大部分像素都是背景，模型很容易学成“全预测背景也不错”。Stage1 patch training 人为提高了裂纹像素出现频率，让模型先建立局部裂纹表征。

但是 patch training 只看局部，容易不知道哪些正常结构在全局上不应该被判为裂纹。所以 Stage2 用完整 ROI fine-tune，让模型把局部裂纹特征放回完整图像上下文中理解。

### 18.2 为什么 Stage1 要有多种 positive patch

裂纹可能在图像中间，也可能贴近 crop 边缘；可能小，也可能长；可能单独出现，也可能和复杂背景贴在一起。`positive_center/shift/context/boundary` 让同一条裂纹以不同位置和尺度出现，提高模型对位置和尺度变化的鲁棒性。

### 18.3 为什么要有 near-miss negative 和 hard negative

普通 normal patch 太容易。模型真正容易错的是“很像裂纹但不是裂纹”的局部纹理。near-miss negative 来自裂纹附近但不含正像素的区域，hard negative 来自 crack 图像中远离裂纹的背景区域。这两类负样本能让模型学习更细的决策边界。

### 18.4 为什么 Stage2 需要 hard-normal replay

部署时最危险的一类错误是 normal 图上的 false positive。Stage2 的 hard-normal replay 相当于在线困难负样本挖掘：模型先暴露自己在哪些 normal 图上误检，再把这些图加入后续训练。这样可以把训练重点从“随机正常背景”转移到“当前模型真的会误判的正常背景”。

### 18.5 为什么需要 OOF global post-processing

固定 0.5 阈值不一定最适合稀疏裂纹分割。太低会增加 normal false positives，太高会漏掉细裂纹。min_area 过滤能去掉小的孤立噪声，但过大又可能删掉真实小裂纹。

所以工程先用每个 fold 的 held-out validation 图生成 OOF probability maps，再统一搜索 threshold/min_area。这样得到的是一个跨 fold 的全局后处理设置，比在单个 fold 上调阈值更稳。

---

## 19. 原模型配置摘要

原始最佳 baseline 主要对应：

```text
Stage1:
  configs/stage1_p0_a40.yaml

Stage2:
  configs/stage2_p0_a40_e50_bs48_pw12.yaml
```

Stage1 关键设置：

| 参数 | 值 |
| --- | ---: |
| `patch_out_size` | 384 |
| `batch_size` | 128 |
| `epochs` | 30 |
| `pretrained` | true |
| `pos_weight` | 12 |
| `freeze_encoder_epochs` | 3 |
| `stage1_sampler_mode` | balanced |
| `stage1_positive_ratio` | 0.50 |
| `stage1_defect_negative_ratio` | 0.25 |
| `stage1_normal_ratio` | 0.25 |
| `stage1_use_replay` | true |

Stage2 关键设置：

| 参数 | 值 |
| --- | ---: |
| `image_size` | 640 |
| `batch_size` | 48 |
| `epochs` | 50 |
| `pretrained` | false |
| `stage1_checkpoint_template` | Stage1 fold best checkpoint |
| `pos_weight` | 12 |
| `random_normal_k_factor` | 1.0 |
| `use_hard_normal_replay` | true |
| `stage2_hard_normal_ratio` | 0.40 |
| `hard_normal_max_repeats_per_epoch` | 2 |
| `target_normal_fpr` | 0.10 |
| `lambda_fpr_penalty` | 2.0 |
| `threshold_grid` | 0.10 to 0.95 step 0.02 |
| `min_area_grid` | [0, 8, 16, 24, 32, 48] |

---

## 20. 工程输出文件如何理解

### 20.1 manifest 和 mask

```text
generated_masks/
manifests/samples.csv
manifests/samples_summary.json
manifests/stage1_fold*_train_index.csv
manifests/stage1_fold*_val_index.csv
manifests/stage1_patch_summary.json
```

这些是数据准备和索引文件。

### 20.2 Stage1 outputs

每个 fold：

```text
stage1/fold<k>/
|- best_stage1.pt
|- last_stage1.pt
|- history.csv
`- replay/
   |- replay_latest.csv
   |- replay_epochXXX.csv
   `- replay_latest_summary.json
```

`best_stage1.pt` 是 Stage2 初始化要用的 checkpoint。

### 20.3 Stage2 outputs

每个 fold：

```text
stage2/fold<k>/
|- best_stage2.pt
|- last_stage2.pt
|- history.csv
|- val_metrics.json
|- val_per_image.csv
|- val_postprocess_search.csv
`- hard_normal/
   |- hard_normal_latest.csv
   |- hard_normal_epochXXX.csv
   `- hard_normal_latest_summary.json
```

### 20.4 OOF outputs

```text
stage2/oof_global_postprocess.json
stage2/oof_global_postprocess_search.csv
stage2/oof_per_image.csv
```

这是最终全局 threshold/min_area 的来源。

### 20.5 Holdout outputs

```text
stage2/holdout_ensemble/
|- prob_maps/
|- raw_binary_masks/
|- masks/
|- inference_summary.csv
|- holdout_metrics.json
`- holdout_per_image.csv
```

`masks/` 是最终后处理后的二值预测结果。

---

## 21. 论文中可以怎样讲这个原模型

可以把方法写成一个“从局部到全局”的两阶段分割框架：

1. **Sparse target problem.** 接触线裂纹在 ROI 图像中占比极小，直接全图训练容易被背景主导。
2. **Stage1 local representation learning.** 通过基于 annotation component 的 patch mining，构造 positive、near-miss negative、hard negative 和 normal negative patch，使模型高频看到裂纹和困难背景。
3. **Stage2 contextual adaptation.** 将 Stage1 学到的 fully convolutional weights 迁移到完整 ROI 图像上，使用 crack/normal 全图训练来恢复全局上下文。
4. **Hard-normal replay.** 在线挖掘 normal 图像中的 false-positive cases，把它们作为困难负样本加入后续 epoch，以降低正常图像误检。
5. **Fold-level validation and OOF post-processing.** 使用 4-fold OOF predictions 选择全局 threshold/min_area，避免只在单一 fold 或 holdout 上调参。
6. **Holdout inference.** 最终使用 4-fold ensemble probability map 和固定 OOF post-processing，在 holdout 上生成预测 mask。


设计后续实验时，要按数据流位置分层控制变量：

| 实验层级 | 可以动的参数 | 必须固定的东西 |
| --- | --- | --- |
| 数据划分实验 | `test_ratio`, `n_folds`, split seed | 不和旧划分直接比最终数值，只比趋势 |
| Stage1 patch 实验 | crop sizes, patch counts, replay ratio, sampler ratios | Stage2 config、fold、epoch、postprocess grid |
| Stage2 loss 实验 | `pos_weight`, `normal_fp_loss_weight`, auxiliary loss weights | Stage1 checkpoint、Stage2 sampling、threshold grid |
| Stage2 sampling 实验 | `random_normal_k_factor`, `stage2_hard_normal_ratio`, hard cap | loss、Stage1 checkpoint、postprocess grid |
| 架构实验 | transformer/attention/deep supervision/boundary head | 数据划分、Stage1 checkpoint、optimizer、epochs、postprocess grid |
| 后处理实验 | threshold grid, `min_area_grid`, target FPR | 固定模型 checkpoint，不重新训练 |

如果比较 two-stage 和 single-stage，核心差别应该只放在初始化路径上：

$$
\text{two-stage}: \theta_{stage2}^{0}=\theta_{stage1}^{best}
$$

$$
\text{single-stage}: \theta_{stage2}^{0}=\theta_{ImageNet}\;\text{or random init}
$$

其他如 split、image size、backbone、loss、epochs、OOF post-processing 都应保持一致。

一句话版本：

```text
The baseline system is a two-stage ResNet34-U-Net segmentation pipeline. The first stage learns crack-sensitive local features from mined patches, while the second stage transfers these features to full ROI images and uses normal-image hard-negative replay to suppress false positives. Final binary masks are produced using a global threshold and connected-component area filter selected from pooled out-of-fold validation predictions.
```

---

## 22. 和后续实验的边界

本文档只覆盖原始两阶段 baseline。后续实验中新增的东西不属于原模型：

| 后续实验 | 是否属于本文档 |
| --- | --- |
| `01_tbn_d1` Transformer bottleneck | no |
| `02_tbn_d1_hnproto` hard-negative prototype cross-attention | no |
| `03_skipgate_d4d3` skip gate | no |
| deep supervision ablation | no |
| boundary auxiliary ablation | no |
| YOLO detection baseline | no |

如果论文写“本文提出的方法”时要包含 `01_tbn_d1`，那么需要在本 baseline 文档基础上另写一节“architectural extension”。如果论文只把 attention 作为对比实验，则原模型就是这里描述的两阶段 ResNet34-U-Net baseline。
