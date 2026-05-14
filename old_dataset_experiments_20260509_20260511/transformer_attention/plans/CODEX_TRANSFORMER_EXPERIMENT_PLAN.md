# Codex 实验方案：ResNet34U-Net + Transformer/Attention 三个创新点

> 目标读者：Codex / 工程实现者。  
> 项目：`KTH-YANGYI/ResNET34U-NET`。  
> 实验资源：4 张 A100。  
> 范围限制：本计划只做 3 个方向：
>
> 1. Transformer bottleneck；
> 2. hard-negative prototype cross-attention；
> 3. decoder skip attention gate。
>
> 不做 SAM、Swin-Unet、SegFormer、YOLO 新实验，除非后续单独立项。

---

## 0. 核心结论

当前指标体系**不是错的**：Dice、IoU、defect image recall、normal FPR 都适合这个任务。问题在于目前还不够“审稿友好”：

1. 需要把 `stage2_score` 从“最终指标”降级为“训练/阈值选择分数”。它里面的 `target_normal_fpr` 和 `lambda_fpr_penalty` 是人为设定的，不适合作为论文唯一主指标。
2. 需要增加统计检验和置信区间。单个 OOF Dice 从 `0.756876` 到 `0.760000` 不能直接说有提升。
3. 需要补充 error decomposition：pixel precision、pixel recall、AUPRC、normal FP count、normal FP area、component-level recall 或 boundary F1。否则审稿人可能问：Dice 提升到底来自少漏检、少误检，还是阈值/后处理凑出来的。
4. 最终 claim 应该基于**同一 split、同一 folds、同一 threshold/min_area 选择规则、同一随机种子策略、paired comparison**。

结论：**指标不用推倒重来，但必须补强。**

---

## 1. 当前项目和 baseline 审计

### 1.1 当前训练流程

当前项目是 two-stage defect segmentation：

```text
Stage1: patch-level model on cropped positive/negative patches
Stage2: load best Stage1 checkpoint, train full-image segmentation model
OOF validation: search one global threshold/min_area
Holdout: reuse selected post-processing parameters
```

当前数据角色：

```text
crack  -> 有 LabelMe mask，用于训练/验证/有标签 holdout
normal -> 无缺陷，训练时全 0 mask，用于训练/验证/有标签 holdout
broken -> unlabeled holdout，只能做定性推理，不允许做 quantitative metric
```

重要约束：

```text
split == trainval -> Stage1/Stage2 训练和 CV validation
split == holdout  -> 不参与训练和 OOF 阈值选择
```

### 1.2 当前最强 U-Net baseline

当前最强配置：

```text
configs/stage2_p0_a40_e50_bs48_pw12.yaml
image_size: 640
batch_size: 48
epochs: 50
pos_weight: 12.0
use_hard_normal_replay: true
threshold_grid: 0.10..0.95 step 0.02
min_area_grid: [0, 8, 16, 24, 32, 48]
```

当前报告的 OOF 结果：

| Variant | Dice | IoU | defect recall | normal FPR | normal FP | threshold | min_area |
|---|---:|---:|---:|---:|---:|---:|---:|
| pw12 baseline | 0.756876 | 0.633878 | 0.995261 | 0.000000 | 0 | 0.80 | 24 |

这应该作为所有 Transformer/Attention 实验的 baseline。

### 1.3 当前模型结构

当前 `src/model.py` 是：

```text
ResNet34 encoder
  x0 = stem
  x1 = layer1
  x2 = layer2
  x3 = layer3
  x4 = layer4
center = ConvBlock(512, 512)
U-Net decoder with skip concat
segmentation_head
```

当前 decoder 是直接：

```python
x = interpolate(x, skip.shape[-2:])
x = torch.cat([x, skip], dim=1)
x = ConvBlock(x)
```

这说明三个创新点都有明确插入位置：

```text
Transformer bottleneck -> center 后面或替代 center
Prototype cross-attention -> x4/center tokens
Skip attention gate -> decoder4/decoder3 的 skip 分支
```

---

## 2. 指标审计：现在的指标是什么，需不需要大改

### 2.1 当前代码里的指标定义

当前 `src/metrics.py` 中已经有：

```text
Dice = 2TP / (2TP + FP + FN)
IoU  = TP / (TP + FP + FN)
```

并在 `evaluate_prob_maps()` 中做：

```text
probability map -> threshold -> connected component min_area filtering -> binary pred mask
```

然后按图像类型汇总：

```text
defect_dice          = mean Dice over defect images only
defect_iou           = mean IoU over defect images only
defect_image_recall  = fraction of defect images with any positive prediction
normal_fpr           = fraction of normal images with any positive prediction
normal_fp_count      = number of normal images with any positive prediction
normal_fp_pixel_sum  = total predicted positive pixels on normal images
normal_largest_fp_area_* = false-positive component area statistics
```

`stage2_score` 当前是：

```text
stage2_score = defect_dice - lambda_fpr_penalty * max(0, normal_fpr - target_normal_fpr)
```

当前 `target_normal_fpr=0.10`，`lambda_fpr_penalty=2.0`。

### 2.2 是否需要大改？

不需要大改。建议：

```text
保留：Dice、IoU、defect image recall、normal FPR、normal FP count、normal FP area。
补充：pixel precision/recall、AUPRC、component recall 或 boundary F1、置信区间、paired significance test。
降级：stage2_score 只用于 training checkpoint selection，不作为论文唯一主指标。
```

理由：

1. Dice 和 IoU 是 segmentation overlap 的标准指标，适合像素级缺陷分割。
2. normal FPR 对你的任务非常重要，因为模型不能在正常接触线图像上乱报 defect。
3. defect image recall 也重要，但它现在的定义是“只要预测出任何 positive pixel 就算命中”，太宽松，只能作为 image-level detection 辅助指标，不能替代 Dice/IoU。
4. 你的正类像素非常少，属于严重类别不平衡任务；因此需要 PR/AUPRC，而不是 ROC-AUC 作为 threshold-free 辅助指标。
5. 小数据集必须报告 uncertainty，否则一个很小的提升可能只是随机种子或 fold 波动。

### 2.3 最终推荐指标层级

#### Primary endpoint

主指标建议定义为：

```text
Primary metric:
  OOF macro defect Dice at the globally selected threshold/min_area

Hard constraints:
  normal_fp_count <= baseline_normal_fp_count + 1
  defect_image_recall >= 0.99

Strict thesis claim:
  normal_fp_count == 0 is preferred because current baseline is already 0.
```

当前 baseline 的 `normal_fp_count=0`，所以任何新方法如果 Dice 提高但 normal FP 明显变多，不能说是“总体提升”，只能说“segmentation-overlap improved under more false alarms”。

#### Secondary endpoints

```text
1. defect_iou
2. pixel_precision_defect
3. pixel_recall_defect
4. pixel_f1_defect
5. pixel_AUPRC_all_labeled
6. defect_component_recall@tolerance
7. boundary_F1@2px or boundary_F1@3px
8. normal_fpr
9. normal_fp_count
10. normal_fp_pixel_sum / p95 / max largest component area
```

#### Final test endpoint

OOF 用于模型选择。最终论文结果还要加一个 frozen holdout evaluation：

```text
1. 用 trainval 4-fold OOF 选择 method、threshold、min_area。
2. 不允许在 holdout 上调 threshold 或 min_area。
3. 用 4-fold ensemble 或固定 fold models 推理 holdout。
4. 只在 labeled holdout crack + normal 上算 quantitative metrics。
5. broken holdout 只能放 qualitative figures，不进 quantitative table。
```

---

## 3. “怎么算有提升”：预注册判定规则

### 3.1 不能这样算

以下情况不能 claim improvement：

```text
只看单次 OOF Dice 高 0.001~0.003。
只调 threshold/min_area 得到更好数值，但没有固定选择规则。
Dice 提高但 normal FP 从 0 增加到多个。
只在某一个 fold 提高。
只在训练验证集上调完所有东西后，再把同一批结果当 final test。
只报告 best seed，不报告 seed variability。
```

### 3.2 可以 claim “有提升”的最低标准

对每个 variant，和 baseline 做 paired comparison。定义：

```text
DeltaDice_i = Dice_i(variant) - Dice_i(baseline)
DeltaIoU_i  = IoU_i(variant)  - IoU_i(baseline)
```

其中 `i` 是同一张 defect validation image，baseline 和 variant 必须来自同一个 fold、同一个 OOF protocol。

一个 variant 可以被称为 **clear improvement**，需要同时满足：

```text
A. OOF macro Dice 提升 >= +0.010
B. paired bootstrap 95% CI for DeltaDice 的 lower bound > 0
C. normal_fp_count <= baseline_normal_fp_count + 1
D. defect_image_recall >= 0.99
E. 至少 2/3 个 confirmation seeds 的 Dice 高于对应 baseline seed
F. final labeled holdout 上没有明显退化：DeltaDice >= 0 且 normal_fp_count 不明显恶化
```

如果只满足 A，但不满足 B：

```text
报告为 trend，不要写 statistically significant improvement。
```

如果 Dice 提升但 normal FP 增加：

```text
报告为 overlap improvement with higher false-alarm risk。
不要写 overall best。
```

如果 Dice 提升不到 +0.010，但 CI 明显大于 0：

```text
可以写 small but statistically supported improvement。
不建议作为论文主贡献，除非 qualitative error analysis 很强。
```

### 3.3 统计检验要求

Codex 需要实现一个比较脚本：

```text
scripts/compare_oof_variants.py
```

输入：

```text
baseline per-image CSV
variant per-image CSV
```

输出：

```text
mean_delta_dice
95% paired bootstrap CI for delta_dice
paired permutation p-value for delta_dice
mean_delta_iou
95% paired bootstrap CI for delta_iou
normal_fp_count difference
McNemar or exact paired test for normal FP indicator
component/boundary metric deltas if available
```

bootstrap 要求：

```text
n_bootstrap = 10000
resampling unit = image, not pixel
stratify by sample_type and fold if possible
random_seed fixed, e.g. 20260510
```

为什么按 image bootstrap，而不是 pixel bootstrap：

```text
同一张图内像素高度相关，把像素当独立样本会夸大显著性。
```

---

## 4. Codex 实现任务

### 4.1 新增统一 model builder

当前问题：多个脚本里直接调用：

```python
model = build_model(pretrained=False)
```

对于新 architecture，这会导致 OOF search / holdout inference 构建成 baseline 模型，从而无法加载新 checkpoint。

Codex 必须新增：

```python
def build_model_from_config(cfg):
    return build_model(
        pretrained=bool(cfg.get("pretrained", False)),
        deep_supervision=bool(cfg.get("deep_supervision_enable", False)),
        boundary_aux=bool(cfg.get("boundary_aux_enable", False)),
        transformer_bottleneck_enable=bool(cfg.get("transformer_bottleneck_enable", False)),
        transformer_bottleneck_layers=int(cfg.get("transformer_bottleneck_layers", 0)),
        transformer_bottleneck_heads=int(cfg.get("transformer_bottleneck_heads", 8)),
        transformer_bottleneck_dropout=float(cfg.get("transformer_bottleneck_dropout", 0.1)),
        prototype_attention_enable=bool(cfg.get("prototype_attention_enable", False)),
        prototype_bank_path=cfg.get("prototype_bank_path", ""),
        skip_attention_enable=bool(cfg.get("skip_attention_enable", False)),
        skip_attention_levels=cfg.get("skip_attention_levels", ["d4", "d3"]),
    )
```

然后替换以下脚本中的 direct `build_model()`：

```text
scripts/train_stage2.py
scripts/search_oof_postprocess.py
scripts/infer_holdout.py
scripts/evaluate_val.py
任何会 load best_stage2.pt 的脚本
```

### 4.2 checkpoint loading 兼容

Transformer/attention 模块是 Stage2 新增模块，而 Stage1 checkpoint 里没有这些参数。当前 `load_checkpoint(..., strict=True)` 会报错。

Codex 需要实现：

```python
def load_compatible_stage1_checkpoint(path, model, map_location):
    checkpoint = torch.load(path, map_location=map_location)
    state = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    compatible = {
        k: v for k, v in state.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print loaded/missing/unexpected counts
    return checkpoint
```

Config flag：

```yaml
stage1_load_strict: false
```

对 baseline 保持：

```yaml
stage1_load_strict: true
```

对所有 Transformer/attention variants：

```yaml
stage1_load_strict: false
```

---

## 5. 创新点 1：Transformer Bottleneck

### 5.1 目的

在 ResNet34 layer4 后得到低分辨率 feature map：

```text
input 640x640 -> x4 about 20x20, channels=512
20x20 = 400 tokens
```

这里加 Transformer self-attention 成本可控，而且不会破坏 U-Net decoder 的定位能力。

### 5.2 模块设计

新增到 `src/model.py` 或 `src/transformer_blocks.py`：

```python
class TransformerBottleneck(nn.Module):
    def __init__(self, channels=512, num_layers=2, num_heads=8,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.pos = nn.Conv2d(channels, channels, kernel_size=3,
                             padding=1, groups=channels)
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=int(channels * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_pos = x + self.pos(x)
        tokens = x_pos.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        y = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x + self.gamma * y
```

`gamma` 零初始化是为了避免 Stage2 一开始破坏从 Stage1 加载来的 CNN 表征。

### 5.3 Configs

```yaml
model_name: resnet34_unet_tbn_d1
transformer_bottleneck_enable: true
transformer_bottleneck_layers: 1
transformer_bottleneck_heads: 8
transformer_bottleneck_dropout: 0.1
stage1_load_strict: false
```

```yaml
model_name: resnet34_unet_tbn_d2
transformer_bottleneck_enable: true
transformer_bottleneck_layers: 2
transformer_bottleneck_heads: 8
transformer_bottleneck_dropout: 0.1
stage1_load_strict: false
```

---

## 6. 创新点 2：Hard-Negative Prototype Cross-Attention

### 6.1 目的

项目已有 hard-negative mining / hard-normal replay。这个创新点要把已有 pipeline 的强项和 Transformer 连接起来，而不是简单换 backbone。

核心想法：

```text
Stage1 patch mining 找到 positive / near-miss negative / hard negative / normal negative patch
    -> 提取 patch-level bottleneck features
    -> 构造 positive 和 negative prototype memory bank
Stage2 full-image bottleneck tokens
    -> query prototypes by cross-attention
    -> 获得 hard-negative-aware global context
    -> 再送入 U-Net decoder
```

### 6.2 Prototype bank 构建脚本

新增：

```text
scripts/build_stage1_prototype_bank.py
```

输入：

```text
--config configs/stage2_xxx.yaml
--fold 0/1/2/3
```

读取：

```text
stage1_checkpoint
manifests/stage1_fold{fold}_train_index.csv
```

输出：

```text
outputs/prototype_banks/{experiment_name}/fold{fold}/prototype_bank.pt
```

`prototype_bank.pt` 内容：

```python
{
    "pos_prototypes": FloatTensor[N_pos, 512],
    "neg_prototypes": FloatTensor[N_neg, 512],
    "meta": {
        "fold": fold,
        "stage1_checkpoint": path,
        "pos_patch_types": [...],
        "neg_patch_types": [...],
        "feature_layer": "center_or_layer4_gap",
        "normalize": True,
    }
}
```

Patch 类型建议：

```text
positive:
  positive_boundary
  positive_shift
  positive_center
  positive_context

negative:
  near_miss_negative
  hard_negative
  normal_negative
  replay_defect_negative_fp
  replay_normal_negative_fp
```

如果某些 replay 类型不存在，不报错，跳过。

Prototype 抽取方法：

```text
1. 用 Stage1 best checkpoint 构建 baseline ResNet34-U-Net。
2. 注册 hook 到 encoder_layer4 或 center 输出。
3. 对每个 patch 做 forward。
4. 对 feature map 做 global average pooling -> 512-d vector。
5. L2 normalize。
6. 每类最多保留 K 个 prototype，例如 K=128；如果数量太大，按固定 seed reservoir sampling。
```

默认：

```yaml
prototype_pos_max: 128
prototype_neg_max: 128
prototype_l2_normalize: true
```

### 6.3 Cross-attention 模块

新增：

```python
class PrototypeCrossAttention(nn.Module):
    def __init__(self, channels=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(channels)
        self.norm_out = nn.LayerNorm(channels)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.type_embed = nn.Parameter(torch.zeros(2, channels))

    def forward(self, x, pos_proto, neg_proto):
        b, c, h, w = x.shape
        q = x.flatten(2).transpose(1, 2)
        q = self.norm_q(q)

        proto = torch.cat([
            pos_proto + self.type_embed[0],
            neg_proto + self.type_embed[1],
        ], dim=0)
        proto = proto.unsqueeze(0).expand(b, -1, -1)

        out, attn = self.mha(q, proto, proto, need_weights=False)
        out = self.norm_out(out)
        y = out.transpose(1, 2).reshape(b, c, h, w)
        return x + self.gamma * y
```

### 6.4 Optional auxiliary loss

第一版可以只做 residual cross-attention，不做辅助损失，降低风险。

第二版再加：

```yaml
prototype_aux_weight: 0.02
```

辅助 loss 思路：

```text
1. 将 GT mask 下采样到 bottleneck resolution。
2. bottleneck token 与 pos/neg prototype 做 cosine similarity。
3. mask-positive token 应该更接近 pos prototypes。
4. mask-negative token 应该更接近 neg prototypes。
```

不建议一开始就把辅助 loss 权重大于 `0.05`，否则可能压过 BCE+Dice 主任务。

---

## 7. 创新点 3：Decoder Skip Attention Gate

### 7.1 目的

当前 decoder 直接 concat skip feature。对于细小裂纹/接触线缺陷，浅层 skip feature 很容易把纹理噪声带进 decoder，造成 false positives。

目标：

```text
decoder feature = query / gating signal
encoder skip feature = local detail
attention gate = decide which skip regions should pass
```

### 7.2 推荐先做低风险版本

先实现 Attention U-Net 风格 additive gate，而不是直接在高分辨率 feature 上做 full self-attention。

新增：

```python
class SkipAttentionGate(nn.Module):
    def __init__(self, skip_channels, gate_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(16, min(skip_channels, gate_channels) // 2)
        self.skip_proj = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.gate_proj = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, skip, gate):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        a = self.psi(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * (1.0 + self.gamma * (a - 1.0))
```

使用位置：

```text
decoder4: gate x3 skip, because x3 resolution about 40x40
decoder3: gate x2 skip, because x2 resolution about 80x80
```

不建议第一版 gate `x1` 和 `x0`：

```text
x1/x0 resolution high，容易增加显存和训练不稳定。
```

Config：

```yaml
skip_attention_enable: true
skip_attention_levels: ["d4", "d3"]
skip_attention_type: additive
stage1_load_strict: false
```

### 7.3 Optional window cross-attention

如果 additive gate 明显有效，再做 window-MHA 版本。不要第一版直接做全局 MHA：

```text
80x80 = 6400 tokens
full attention O(N^2) 显存/计算不划算
```

---

## 8. 实验矩阵

### 8.1 Phase 0：工程 smoke test

每个 variant 先跑：

```text
fold 0
1 epoch
batch_size 2 or 4
amp true
```

检查：

```text
forward/backward 不报错
Stage1 checkpoint 能兼容加载
best_stage2.pt 能保存
OOF/evaluate/infer 脚本能用 cfg 构建同 architecture
输出 mask shape 正确
history.csv 有所有指标
```

### 8.2 Phase 1：single-seed 4-fold OOF screening

固定：

```text
seed: 42
folds: 0,1,2,3
same trainval split
same Stage1 checkpoints
same image_size=640
same effective batch size
same pos_weight=12
same hard normal replay settings
same OOF threshold/min_area search grid
```

实验表：

| ID | Variant | Modules | Purpose |
|---|---|---|---|
| B0 | baseline rerun | current ResNet34-U-Net | 重新跑 baseline，避免旧结果和新代码不可比 |
| T1 | TBN-d1 | Transformer bottleneck depth=1 | 低风险全局上下文 |
| T2 | TBN-d2 | Transformer bottleneck depth=2 | 稍强全局上下文 |
| H1 | HNProto | TBN-d1 + prototype cross-attention | 检验 hard-negative memory 是否有效 |
| S1 | SkipGate | additive skip attention d4+d3 | 检验 skip noise suppression |
| F1 | Full | TBN-d1 + HNProto + SkipGate | 三个创新点组合 |

注意：

```text
H1 和 F1 需要先为每个 fold 构建 prototype_bank.pt。
```

### 8.3 Phase 2：confirmation seeds

从 Phase 1 选：

```text
baseline B0
best single module among T1/T2/H1/S1
full model F1 if Phase 1 不差
```

Seeds：

```text
42
2026
3407
```

判定：

```text
至少 2/3 seeds 的 OOF Dice 高于对应 baseline seed。
均值提升 >= +0.010 优先。
paired bootstrap CI lower > 0 优先。
normal_fp_count 不明显恶化。
```

### 8.4 Phase 3：frozen labeled holdout evaluation

流程：

```text
1. 用 Phase 1/2 的 OOF 结果选择最终 method。
2. 固定 threshold/min_area，不许看 holdout 调参。
3. 用 4-fold models 对 holdout crack+normal 推理。
4. 如果做 ensemble，平均 4 个 fold probability maps 后再 threshold。
5. crack+normal holdout 算 quantitative metrics。
6. broken holdout 只保存 qualitative predictions。
```

新增脚本：

```text
scripts/evaluate_holdout_quant.py
```

输入：

```text
--config final_config.yaml
--global-postprocess outputs/.../oof_global_postprocess.json
--folds 0,1,2,3
--ensemble mean
```

输出：

```text
holdout_metrics.json
holdout_per_image.csv
holdout_qualitative_index.csv
```

---

## 9. 4×A100 运行方式

当前代码不需要 DDP。最稳妥方式：**一张 A100 跑一个 fold**。

示例 runner：

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG="$1"
EXP_NAME="$2"

for FOLD in 0 1 2 3; do
  GPU=$FOLD
  CUDA_VISIBLE_DEVICES=$GPU python scripts/train_stage2.py \
    --config "$CONFIG" \
    --fold "$FOLD" \
    > "outputs/experiments/${EXP_NAME}_fold${FOLD}.log" 2>&1 &
done

wait

python scripts/search_oof_postprocess.py \
  --config "$CONFIG" \
  --folds 0,1,2,3 \
  > "outputs/experiments/${EXP_NAME}_oof_search.log" 2>&1
```

Prototype bank 也按 fold 并行：

```bash
for FOLD in 0 1 2 3; do
  GPU=$FOLD
  CUDA_VISIBLE_DEVICES=$GPU python scripts/build_stage1_prototype_bank.py \
    --config "$CONFIG" \
    --fold "$FOLD" \
    > "outputs/experiments/${EXP_NAME}_proto_fold${FOLD}.log" 2>&1 &
done
wait
```

显存策略：

```text
优先保持 batch_size=48，因为 baseline 是这个 batch size。
如果 transformer variant OOM：
  1. 先把 transformer layers 从 2 降到 1。
  2. 再把 heads 从 8 降到 4。
  3. 最后才降低 batch_size。
如果降低 batch_size，则 baseline rerun 也要用同样 effective batch size，否则不公平。
```

---

## 10. 必须新增/修改的文件清单

### 10.1 Model code

```text
src/model.py
src/transformer_blocks.py      # optional new file
src/prototype_memory.py        # prototype bank loading and attention
src/attention_gates.py         # skip attention gate
```

### 10.2 Training/evaluation scripts

```text
scripts/train_stage2.py
scripts/search_oof_postprocess.py
scripts/evaluate_val.py
scripts/infer_holdout.py
scripts/build_stage1_prototype_bank.py       # new
scripts/compare_oof_variants.py             # new
scripts/evaluate_holdout_quant.py           # new
scripts/run_stage2_variant_4gpu.sh          # new
scripts/run_transformer_experiments_4gpu.sh # new
```

### 10.3 Configs

```text
configs/transformer/stage2_pw12_baseline_seed42.yaml
configs/transformer/stage2_tbn_d1_seed42.yaml
configs/transformer/stage2_tbn_d2_seed42.yaml
configs/transformer/stage2_hnproto_seed42.yaml
configs/transformer/stage2_skipgate_seed42.yaml
configs/transformer/stage2_full_seed42.yaml
```

For confirmation seeds:

```text
configs/transformer/seed2026/*.yaml
configs/transformer/seed3407/*.yaml
```

### 10.4 Outputs

每个实验必须保存：

```text
best_stage2.pt
last_stage2.pt
history.csv
val_per_image.csv
oof_global_postprocess.json
oof_global_postprocess_search.csv
oof_per_image.csv
compare_to_baseline.json
compare_to_baseline.md
```

---

## 11. 新增指标的实现细节

### 11.1 Pixel precision / recall

新增函数：

```python
def pixel_confusion(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    return tp, fp, fn
```

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2TP / (2TP + FP + FN)
```

Report both:

```text
macro average over defect images
micro pooled over all labeled pixels
```

### 11.2 Pixel AUPRC

实现 threshold-free PR metric：

```text
y_true = flattened GT masks over labeled crack+normal images
y_score = flattened probability maps
Average Precision / AUPRC computed from sorted scores
```

不要用 ROC-AUC 作为主指标，因为 positive pixels 很少，PR curve 对不平衡更直观。

### 11.3 Component-level recall

针对 crack/defect 是细小连通结构，增加：

```text
GT connected component is detected if:
  overlap(pred_dilated, gt_component) > 0
or
  IoU(pred_component, gt_component) >= 0.1
```

推荐第一版：

```text
component_recall@3px_dilation
component_precision@3px_dilation
component_F1@3px_dilation
```

注意：

```text
component metric 只作为 secondary metric。
不要用它替代 Dice/IoU，因为 GT polygon 的连通性可能受标注细节影响。
```

### 11.4 Boundary F1

如果实现，使用：

```text
boundary_F1@2px
boundary_F1@3px
```

做法：

```text
boundary(mask) = mask XOR erode(mask)
pred boundary matched if within tolerance pixels of GT boundary
GT boundary matched if within tolerance pixels of pred boundary
boundary precision / recall / F1
```

Boundary F1 适合作为论文中“thin structure localization”的辅助指标，但不要把它作为唯一主指标。

---

## 12. 论文表格建议

### 12.1 Main OOF table

| Method | Dice | 95% CI | IoU | defect recall | normal FP | normal FPR | AUPRC | threshold | min_area |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | | | | | | | | | |
| +TBN | | | | | | | | | |
| +HNProto | | | | | | | | | |
| +SkipGate | | | | | | | | | |
| Full | | | | | | | | | |

### 12.2 Paired comparison table

| Variant | Delta Dice | 95% CI | p-value | Delta IoU | normal FP delta | Verdict |
|---|---:|---:|---:|---:|---:|---|
| +TBN | | | | | | |
| +HNProto | | | | | | |
| +SkipGate | | | | | | |
| Full | | | | | | |

### 12.3 Seed stability table

| Variant | Seed 42 | Seed 2026 | Seed 3407 | Mean | Std | Wins vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | | | | | | |
| Best variant | | | | | | |

### 12.4 Final holdout table

| Method | Holdout Dice | IoU | defect recall | normal FP | normal FPR | Notes |
|---|---:|---:|---:|---:|---:|---|
| Baseline | | | | | | labeled crack+normal only |
| Best variant | | | | | | threshold frozen from OOF |

### 12.5 Qualitative figures

每个方法选：

```text
3 true positive examples
3 false negative / under-segmentation examples
3 false positive normal examples if any
3 broken holdout qualitative examples, clearly labeled as unlabeled qualitative only
```

---

## 13. 审稿风险和规避

### 13.1 指标选择被质疑

规避：

```text
引用 Metrics Reloaded 和 Taha & Hanbury。
说明任务同时有 pixel-level segmentation 和 image-level false alarm 风险。
因此报告 overlap metrics + image-level FPR + PR/AUPRC + confidence intervals。
```

### 13.2 normal FPR 为 0 被质疑样本太少

规避：

```text
报告 normal_count 和 normal_fp_count。
报告 confidence interval。
不要只写 0.000000。
```

当前 OOF normal count 约 59，`1/59 = 0.01695`，所以 normal FPR 的分辨率很粗。必须报告 count。

### 13.3 threshold/min_area 泄漏

规避：

```text
OOF 上选 threshold/min_area。
holdout 上只应用 frozen threshold/min_area。
任何 holdout 调参都禁止。
```

### 13.4 只跑一个 seed 被质疑

规避：

```text
screening 可以一个 seed。
最终 claim 必须 3 seeds 或至少 baseline+best variant 3 seeds。
```

### 13.5 architecture 改动太多，无法解释贡献

规避：

```text
必须有 ablation：Baseline, +TBN, +HNProto, +SkipGate, Full。
不要只报告 Full。
```

### 13.6 Prototype memory 被质疑数据泄漏

规避：

```text
fold k 的 prototype bank 只能用 fold k 的 training patches。
不能用 validation patches。
不能用 holdout。
```

### 13.7 broken 图像误用

规避：

```text
broken 是 unlabeled holdout，只能 qualitative。
不能计算 Dice/IoU。
不能用 SAM/伪标签训练后再作为 holdout。
```

---

## 14. Codex PR checklist

Codex 完成后，逐项检查：

```text
[ ] Baseline config 仍能 strict load Stage1 checkpoint。
[ ] Transformer/attention configs 能 non-strict compatible load Stage1 checkpoint。
[ ] train_stage2.py、search_oof_postprocess.py、infer_holdout.py 都使用 build_model_from_config(cfg)。
[ ] 每个 variant fold0 smoke test 通过。
[ ] OOF per-image CSV 能按 image_name/sample_id 与 baseline 对齐。
[ ] compare_oof_variants.py 输出 bootstrap CI 和 p-value。
[ ] Prototype bank 每个 fold 只用该 fold training patches。
[ ] Holdout evaluation 不做 threshold tuning。
[ ] broken holdout 不进入 quantitative metrics。
[ ] README 或 EXPERIMENTS 新增 transformer 实验记录。
```

---

## 15. 参考文献 / 权威依据

1. Maier-Hein et al., **Metrics Reloaded: Recommendations for image analysis validation**, Nature Methods, 2024.  
   作用：说明 metric selection 应该 problem-aware，且 image/object/pixel-level validation 要根据任务指纹选择。

2. Taha & Hanbury, **Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool**, BMC Medical Imaging, 2015.  
   作用：说明 segmentation metric 有 overlap、distance 等不同类别，多目标任务应组合多个不完全冗余指标。

3. Saito & Rehmsmeier, **The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets**, PLOS ONE, 2015.  
   作用：你的任务正类像素稀少，所以 AUPRC 比 ROC-AUC 更合适作为 threshold-free 辅助指标。

4. Chen et al., **TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation**, arXiv, 2021.  
   作用：支持 CNN/U-Net 负责局部细节，Transformer 负责 global context 的 hybrid 设计。

5. Oktay et al., **Attention U-Net: Learning Where to Look for the Pancreas**, arXiv, 2018.  
   作用：支持 attention gate 抑制 irrelevant regions，并可集成到 U-Net skip pathway。

6. Shrivastava et al., **Training Region-Based Object Detectors with Online Hard Example Mining**, CVPR, 2016.  
   作用：支持 hard example mining / hard negative mining 的训练思想；你的 HNProto 是把这种思想从 sampling 扩展到 prototype attention memory。
