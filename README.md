# Diffusion 缺陷检测项目 Baseline

本仓库是我在 **diffusion 扩散生成缺陷检测项目** 中使用的一个 **baseline**。  

当前 baseline 采用的是 **两阶段 U-Net 缺陷分割方案**：

1. `stage1`：先在局部缺陷 patch 上训练模型，让网络先学会缺陷纹理和边界。
2. `stage2`：再把 `stage1` 学到的权重迁移到整图 ROI 分割，输出完整缺陷 mask。

这个 baseline 的意义是：

- 给后续的 diffusion 缺陷生成、合成样本增强、伪缺陷构造提供稳定对照组。
- 先验证当前数据集、划分方式、训练脚本和评估指标是否可靠。
- 后续无论加入 diffusion augmentation、diffusion pretraining，还是直接做生成式缺陷合成，都可以和这个 baseline 做公平对比。


## 1. 项目目标

当前仓库主要解决以下问题：

- 整理现有缺陷图、正常图和未标注 holdout 图像。
- 按 `video_id` 做数据划分，尽量避免同视频样本泄漏到训练集和验证集。
- 构建两阶段 U-Net 分割训练流程。
- 在验证集上输出可比较的分割指标。
- 在未标注 holdout 图上输出概率图和二值 mask，为后续人工检查或补标做准备。

## 2. Baseline 方法概览

### 2.1 Stage 1: Patch 级训练

`stage1` 不直接看整张图，而是先训练局部 patch 分割模型。这样做的原因是缺陷区域通常较小，如果直接整图训练，模型容易学成“全背景”。

当前 patch 设计包含三类样本：

- 正样本 patch：从缺陷区域附近裁切，包含 `positive_center`、`positive_shift`、`positive_context`
- 困难负样本 patch：从缺陷图的非缺陷区域裁切，记为 `hard_negative`
- 普通负样本 patch：从正常图裁切，记为 `normal_negative`

其中每张缺陷图默认会生成：

- 1 个 `positive_center`
- 2 个 `positive_shift`
- 1 个 `positive_context`
- 4 个 `hard_negative`

然后再从正常图中采样与 `hard_negative` 数量对齐的 `normal_negative`。

### 2.2 Stage 2: 整图分割微调

`stage2` 会加载 `stage1` 的最优权重，然后在 `640 x 640` 的整图 ROI 上继续训练。  
每个 epoch 中：

- 缺陷训练图全部保留
- 从正常训练图中随机采样与缺陷图数量接近的一部分
- 混合后训练

这个设计的目标是降低类别不平衡带来的偏置。

### 2.3 评估指标

当前验证阶段主要关注以下指标：

- `defect_dice`
- `defect_iou`
- `defect_image_recall`
- `normal_fpr`

其中 `stage2` 选择最佳 checkpoint 时，不是只看单一 Dice，而是优先满足：

1. `normal_fpr <= 0.10`
2. 在满足上面条件的前提下，尽量提高 `defect_dice`

因此这个 baseline 的目标不只是“分得准”，还要控制正常样本上的误报。

## 3. 数据集说明

当前仓库中的数据位于 `dataset_new/`，并已通过脚本整理为 `manifests/` 下的若干 CSV/JSON 清单。

### 3.1 当前数据规模

根据 `manifests/split_summary.json`：

- 已标注缺陷图：`81`
- 未标注缺陷 holdout 图：`42`
- 正常图池：`5414`
- 为未来实验单独预留的 normal future holdout：`2082`
- 交叉验证折数：`4`

### 3.2 划分原则

数据划分不是纯随机图片级切分，而是尽量基于 `video_id` 划分，目的是减少来自同一视频的样本泄漏到训练集和验证集。

这对缺陷检测任务非常重要，因为同一视频中的帧往往高度相似。如果直接随机切图，验证指标会显著偏乐观，无法作为后续 diffusion 方法的可靠对照。

### 3.3 目录中的主要数据来源

- `dataset_new/train/images/`：已标注缺陷图像与对应 JSON
- `dataset_new/train/masks/`：缺陷 mask
- `dataset_new/val/`：当前未标注 holdout 图像
- `dataset_new/normal_crops_selected/`：正常图像
- `dataset_new/图片来源视频映射/`：图像到视频的映射关系

## 4. 项目结构

```text
UNET_two_stage/
├─ configs/                 # stage1 / stage2 配置文件
├─ dataset_new/             # 原始数据
├─ manifests/               # 数据清单、划分结果、patch 索引
├─ outputs/                 # 训练输出、预测结果、可视化产物
├─ scripts/                 # 数据准备、训练、验证、推理脚本
├─ src/                     # 模型、数据集、loss、metrics、trainer
├─ implementation_plan.md
├─ unet_training_plan_for_codex.md
└─ README.md
```

## 5. 核心脚本说明

- `scripts/prepare_dataset.py`
  作用：扫描数据集、挂接 `video_id`、生成 `master_manifest.csv`、训练/验证划分文件等。

- `scripts/build_patch_index.py`
  作用：为 `stage1` 生成 patch 级索引文件，例如 `stage1_fold0_train_index.csv`。

- `scripts/train_stage1.py`
  作用：训练 patch 级分割模型，输出 `best_stage1.pt` 与训练历史。

- `scripts/train_stage2.py`
  作用：加载 `stage1` 最优权重，在整图上继续训练，输出 `best_stage2.pt`。

- `scripts/evaluate_val.py`
  作用：在验证集上评估 `stage2` 模型，并保存 `val_metrics.json` 与逐图结果。

- `scripts/infer_holdout.py`
  作用：对未标注 holdout 图做推理，输出概率图和二值 mask。

## 6. 环境依赖

当前仓库没有单独整理 `requirements.txt`，从代码依赖看，至少需要以下包：

```bash
pip install torch torchvision numpy pillow opencv-python pyyaml python-pptx
```

说明：

- `python-pptx` 只在导出网络结构 PPT 时需要。
- 训练主流程核心依赖是 `torch`、`torchvision`、`numpy`、`Pillow`、`opencv-python`、`PyYAML`。

推荐 Python 版本：`3.10+`

## 7. 快速开始

### 7.1 重新生成数据清单

```bash
python scripts/prepare_dataset.py
```

### 7.2 重新生成 Stage 1 patch 索引

```bash
python scripts/build_patch_index.py
```

### 7.3 训练 Stage 1

```bash
python scripts/train_stage1.py --config configs/stage1.yaml --fold 0
```

输出目录默认是：

```text
outputs/stage1/fold0/
```

主要文件包括：

- `best_stage1.pt`
- `last_stage1.pt`
- `history.csv`

### 7.4 训练 Stage 2

```bash
python scripts/train_stage2.py --config configs/stage2.yaml --fold 0
```

输出目录默认是：

```text
outputs/stage2/fold0/
```

主要文件包括：

- `best_stage2.pt`
- `last_stage2.pt`
- `history.csv`

### 7.5 验证 Stage 2

```bash
python scripts/evaluate_val.py --config configs/stage2.yaml --fold 0
```

会生成：

- `outputs/stage2/fold0/val_metrics.json`
- `outputs/stage2/fold0/val_per_image.csv`

### 7.6 推理未标注 holdout

```bash
python scripts/infer_holdout.py --config configs/stage2.yaml --fold 0
```

会生成：

- `outputs/stage2/fold0/holdout/prob_maps/`
- `outputs/stage2/fold0/holdout/masks/`

## 8. 当前配置

### 8.1 Stage 1 默认配置

- patch 输出尺寸：`384`
- batch size：`16`
- epochs：`30`
- encoder lr：`5e-5`
- decoder lr：`2e-4`
- 预训练 encoder：`true`
- 前 `3` 个 epoch 冻结 encoder

### 8.2 Stage 2 默认配置

- 整图尺寸：`640`
- batch size：`4`
- epochs：`25`
- encoder lr：`2e-5`
- decoder lr：`1e-4`
- 默认阈值：`0.50`
- 默认加载 `outputs/stage1/fold{fold}/best_stage1.pt`

## 9. Baseline 与 Diffusion 研究的关系

这是整个 diffusion 缺陷检测项目的起点，而不是终点。

后续如果要开展 diffusion 方向实验，这个仓库可以承担以下角色：

- 作为没有 diffusion 生成增强时的基准线
- 作为比较 “加入合成缺陷样本前后” 性能变化的参照
- 作为验证数据划分、指标定义、误报控制策略是否合理的稳定底座

未来可在这个 baseline 基础上继续扩展，例如：

- 用 diffusion 生成缺陷样本，扩充 `stage1` 或 `stage2` 训练集
- 用 diffusion 做正常图到缺陷图的可控生成
- 比较真实缺陷训练、真实+生成混合训练、纯生成预训练等方案
- 评估 diffusion 生成数据对 `defect_dice`、`normal_fpr` 的影响

## 10. 当前仓库的边界

需要明确的是，**当前代码仓库本身还没有实现 diffusion 生成模型训练或采样流程**。  
它目前完成的是：

- 数据整理
- 两阶段 U-Net 分割 baseline
- 验证与 holdout 推理

因此如果后续论文标题或研究方向是 “基于 diffusion 的缺陷检测/缺陷生成”，那么这个仓库最准确的描述应该是：

> diffusion 缺陷检测项目中的传统分割 baseline / 对照实验系统

## 11. 输出与复现实验建议

如果你后续要写论文或实验记录，建议至少固定以下内容：

- fold 编号
- 使用的 `stage1.yaml` / `stage2.yaml`
- 训练时的随机种子
- `val_metrics.json` 中的最终指标
- holdout 推理结果目录

这样后面不管接 diffusion augmentation、synthetic defect generation 还是别的生成式方法，都能和当前 baseline 做严格对照。

