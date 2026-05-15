# 金属裂缝 UNet 实验计划（固定 8/1/1 split）

**版本日期**：2026-05-15
**当前入口**：`configs/canonical_baseline.yaml`
**当前数据集**：`/Users/yangyi/Desktop/masterthesis/dataset_crack_normal_unet_811`
**任务**：金属裂缝二分类语义分割，区分 `crack` 与 `normal`，输出裂缝 mask。
**核心目标**：在固定 train/val/test 划分下，先建立一个干净的 canonical baseline，再比较 bottleneck attention、skip attention gate、decoder self-attention、prototype cross-attention 对裂缝召回和 normal false positive 的影响。

---

## 1. 实验原则

这轮实验不再使用 4-fold cross validation。当前项目只保留一个正式入口：

```text
configs/canonical_baseline.yaml
```

所有新实验都从 canonical baseline 派生，只覆盖必要字段：

```text
model_variant
attention 相关参数
loss / replay 相关消融参数
seed
save_dir
```

四个历史实验只作为 frozen experiment set 保存，不再混入当前主模型逻辑。模型结构通过 registry/variant 管理，主训练脚本不再塞多个实验开关。

---

## 2. 当前代码前置条件

正式开跑前先确认以下状态：

```text
[ ] samples.csv 来自 dataset_crack_normal_unet_811
[ ] train/val/test 使用数据集自带固定 split
[ ] active config 中没有 fold / cv_fold / n_folds
[ ] Stage1 输出到 outputs/experiments/<run>/stage1
[ ] Stage2 输出到 outputs/experiments/<run>/stage2
[ ] 每个 seed 有独立 save_dir，避免覆盖
```

当前代码已补齐支撑本文档的关键执行能力：

| 能力 | 用途 | 状态 |
|---|---|---|
| Stage2-only 训练 | 支撑 T1: no Stage1 pretraining | 已支持 `stage1_checkpoint: ""` 时跳过加载 |
| quantitative test eval | 生成 test Dice/IoU/FPR/source-wise 表 | `infer_holdout.py` 输出 `holdout_metrics.json` / `holdout_per_image.csv` |
| prototype bank 构建 | 支撑 M6: prototype cross-attention | 已有脚本和 M6 config，需要按 8/1/1 fixed split 跑一遍 |

---

## 3. 总体顺序

推荐执行顺序：

```text
Step 0: 数据、manifest、config sanity check
Step 1: 跑 M0 baseline 单 seed，确认全流程跑通
Step 2: 在 M0 上做训练机制消融，锁定正式 baseline
Step 3: 在 M0 上做后处理消融，锁定统一 evaluation protocol
Step 4: 跑主模型实验：baseline / TBN / skip gate / self-attention
Step 5: 跑 M6 prototype cross-attention 单 seed，判断是否值得补齐
Step 6: 只对最好的 1-2 个模型做可选扩展
```

一句话原则：

```text
先定 baseline，再比较模型结构；val 选参数，test 只做最终评估。
```

---

## 4. A 组：baseline sanity run

先跑一个最小闭环：

| ID | 模型 | seed | 目的 |
|---|---|---:|---|
| M0-sanity | ResNet34-UNet baseline | 20260515 | 确认数据、mask、loss、val eval、test eval、可视化全部正常 |

检查内容：

```text
loss 是否正常下降
val Dice / IoU 是否合理
normal 图是否大量误检
threshold / min_area search 是否正常
test evaluation 是否只在最终执行
预测 mask 是否和原图对齐
```

如果 M0-sanity 没跑通，不继续跑 attention。

---

## 5. B 组：baseline 训练机制消融

这组只在 `M0: resnet34_unet_baseline` 上跑，用来决定正式 baseline。第一轮每个实验只跑 1 个 seed。

| ID | 实验 | 改动 | 目的 |
|---|---|---|---|
| T0 | full baseline | Stage1 + Stage2 + hard normal replay | 主 baseline |
| T1 | no Stage1 pretraining | Stage2 不加载 Stage1 checkpoint | 判断 two-stage 是否有必要 |
| T2 | no hard normal replay | 关闭 Stage2 hard normal replay | 判断困难 normal 是否降低误检 |
| T3 | normal FP loss | 开启 normal false-positive penalty | 判断能否进一步压低 normal FP |

### 5.1 T1: no Stage1 pretraining

计划配置：

```yaml
stage2:
  stage1_checkpoint: ""
  stage1_load_strict: false
  pretrained: true
```

执行前需要改代码：当 `stage1_checkpoint` 为空时，`train_stage2.py` 应跳过 checkpoint load，直接使用 `pretrained: true` 的 encoder 初始化。

### 5.2 T2: no hard normal replay

```yaml
stage2:
  use_hard_normal_replay: false
  stage2_hard_normal_ratio: 0.0
```

### 5.3 T3: normal FP loss

先试一个保守权重：

```yaml
stage2:
  normal_fp_loss_weight: 0.05
  normal_fp_topk_ratio: 0.10
```

如果 FPR 降低但 recall 基本不掉，再试：

```yaml
stage2:
  normal_fp_loss_weight: 0.10
  normal_fp_topk_ratio: 0.10
```

### 5.4 baseline 锁定规则

| 现象 | 决策 |
|---|---|
| T1 明显差于 T0 | 正式 baseline 保留 Stage1 pretraining |
| T1 和 T0 接近 | 仍可保留 T0，因为 two-stage 是当前项目主流程 |
| T2 normal FPR 明显升高 | 正式 baseline 保留 hard normal replay |
| T3 降低 FPR 但 recall 明显下降 | 不采用或降低权重 |
| T3 降低 FPR 且 Dice/Recall 基本不掉 | 可加入正式 baseline |

最终正式 baseline 只选一个版本，后续所有模型结构都从它派生。

---

## 6. C 组：后处理协议消融

后处理不需要额外训练，只对同一组模型输出重新评估。所有模型必须使用同一套协议。

| ID | threshold | min_area | 目的 |
|---|---:|---:|---|
| P0 | 0.50 | 0 | 看 raw model performance |
| P1 | val selected | 0 | 看 threshold 选择贡献 |
| P2 | val selected | val selected | 看完整后处理贡献 |

推荐正式主表使用：

```text
P2: validation-selected threshold + validation-selected min_area
```

同时在附表报告 P0，说明模型原始输出能力。

搜索范围：

```yaml
stage2:
  threshold_grid_start: 0.10
  threshold_grid_end: 0.95
  threshold_grid_step: 0.02
  min_area_grid: [0, 8, 16, 24, 32, 48]
```

规则：

```text
每个 seed 只在 val set 上选择 threshold / min_area
选好以后固定参数，在 test set 上评估
不能在 test set 上重新选择 threshold / min_area
```

---

## 7. D 组：模型结构主实验

这组是论文/报告主线。只有在正式 baseline 和 postprocess protocol 锁定后再跑。

### 7.1 当前主模型

| ID | model_variant | 名称 | 目的 | 优先级 |
|---|---|---|---|---|
| M0 | `resnet34_unet_baseline` | baseline | 主基线 | 必跑 |
| M1 | `tbn_d1` | bottleneck attention | 看 bottleneck 全局上下文建模是否有效 | 必跑 |
| M2 | `skipgate_d4d3` + `["d4"]` | skip gate d4 | 看高层 skip filtering 是否有效 | 次优先 |
| M3 | `skipgate_d4d3` + `["d3"]` | skip gate d3 | 看中层 skip filtering 是否有效 | 次优先 |
| M4 | `skipgate_d4d3` + `["d4", "d3"]` | skip gate d4+d3 | 看多尺度 skip filtering 是否互补 | 必跑 |
| M5 | `selfattn_d4d3` | decoder self-attention d4+d3 | 和 attention gate 对比，验证 decoder token mixing 是否有效 | 必跑 |
| M6 | `tbn_d1_hnproto` | prototype cross-attention | 让 bottleneck feature cross-attend 到正/负 prototype bank | 第二轮 |

可选模型暂不放入第一轮主表：

| model_variant | 原因 |
|---|---|
| `resnet34_unet_aux` | deep supervision/boundary head 会分散当前 attention 主线 |
| `tbn_skipgate_d4d3` | 当前需要新增组合模型，放在扩展实验 |

### 7.2 M1 配置

```yaml
stage2:
  model_variant: tbn_d1
  transformer_bottleneck_layers: 1
  transformer_bottleneck_heads: 8
  transformer_bottleneck_dropout: 0.1
```

### 7.3 M2 / M3 / M4 配置

M2: d4 only

```yaml
stage2:
  model_variant: skipgate_d4d3
  skip_attention_levels: ["d4"]
  skip_attention_gamma_init: 0.0
```

M3: d3 only

```yaml
stage2:
  model_variant: skipgate_d4d3
  skip_attention_levels: ["d3"]
  skip_attention_gamma_init: 0.0
```

M4: d4 + d3

```yaml
stage2:
  model_variant: skipgate_d4d3
  skip_attention_levels: ["d4", "d3"]
  skip_attention_gamma_init: 0.0
```

### 7.4 M5 配置

```yaml
stage2:
  model_variant: selfattn_d4d3
  stage1_load_strict: false
  self_attention_gamma_init: 0.0
```

解释重点：

```text
skip attention gate: 用 gate 过滤 encoder skip features，更偏向抑制噪声和 normal FP
self-attention: 在 decoder feature map 内部做 token mixing，更偏向建模长程裂缝结构
```

### 7.5 M6 配置

M6 是 prototype cross-attention，不建议和 M1 混为同一个结论。它的差别是：

```text
M1: image tokens 自己做 transformer self-attention
M6: image tokens 作为 query，positive/negative prototypes 作为 key/value
```

需要先用 Stage1 checkpoint 构建 prototype bank：

```bash
python scripts/build_stage1_prototype_bank.py \
  --config configs/experiments/prototype_cross_attention.yaml
```

候选配置：

```yaml
stage2:
  model_variant: tbn_d1_hnproto
  transformer_bottleneck_layers: 1
  transformer_bottleneck_heads: 8
  transformer_bottleneck_dropout: 0.1
  prototype_bank_path: outputs/experiments/811_m6_hnproto_s20260515/prototype_bank.pt
  prototype_attention_heads: 8
  prototype_attention_dropout: 0.1
```

建议先只跑 1 个 seed。如果 M6 明显降低 normal FPR 或提升 hard cases，再补齐 3 seeds。

---

## 8. seed 与输出目录

正式主实验使用 3 seeds：

```text
20260515
20260516
20260517
```

每个模型和 seed 必须有独立输出目录：

```text
outputs/experiments/811_m0_baseline_s20260515/stage1
outputs/experiments/811_m0_baseline_s20260515/stage2
outputs/experiments/811_m5_selfattn_d4d3_s20260515/stage2
```

推荐 Stage1 共享策略：

```text
同一个 seed 先训练一个 M0 Stage1 checkpoint。
同一 seed 下的 M0/M1/M2/M3/M4/M5/M6 Stage2 都加载这个 Stage1 checkpoint。
```

这样模型差异主要来自 Stage2 架构，而不是 Stage1 随机性。

第一轮主实验训练量：

```text
M0, M1, M4, M5 各 3 seeds = 12 runs
M2, M3 各 1 seed = 2 runs
总计 14 个 Stage2 runs
```

如果 M2 或 M3 单独效果很好，再补齐到 3 seeds。

M6 先作为第二轮单 seed，不计入第一轮 14 个 Stage2 runs。

---

## 9. 可选实验

### 9.1 prototype cross-attention 补齐

如果 M6 单 seed 有收益，再补齐：

```text
M6 tbn_d1_hnproto × 3 seeds
```

判断重点：

```text
是否减少 normal false positive
是否提升 hard crack recall
prototype bank 是否稳定
```

### 9.2 bottleneck + skip attention 组合

只有当 M1 或 M4 明显优于 baseline 时，再考虑新增组合模型：

```text
TransformerSkipGateUNet
```

候选配置：

```yaml
stage2:
  model_variant: tbn_skipgate_d4d3
  transformer_bottleneck_layers: 1
  transformer_bottleneck_heads: 8
  transformer_bottleneck_dropout: 0.1
  skip_attention_levels: ["d4", "d3"]
  skip_attention_gamma_init: 0.0
```

如果组合模型只是小幅提升，但复杂度明显增加，放在附表即可。

### 9.3 原图输入协议

当前主实验不做 Stage2 resize，`640x640` 和 `2160x3840` 原图都直接进模型。为了让 mixed-size 数据能正常 batch，训练和评估按原始尺寸分桶：

```yaml
stage2:
  image_size: native
  stage2_native_size: true
  batch_size_by_image_size:
    640x640: 320
    3840x2160: 16
```

如果后续新增其他尺寸，只需要补对应的 `HxW: batch_size`。这里的 `HxW` 是模型实际张量顺序；sampler 也兼容误写成 `WxH`。主论文表应额外报告 image-size-wise metrics，确认高分辨率样本没有被低分辨率样本的结果掩盖。

---

## 10. 不建议第一轮优先跑

| 实验 | 原因 |
|---|---|
| 4-fold / group CV | 当前数据已固定 8/1/1，且 active code 已删除 fold 逻辑 |
| prototype cross-attention full sweep | 可以先跑 M6 单 seed，但不建议第一轮直接 3 seeds |
| aux head / boundary head | 会分散当前 attention 主线 |
| 大规模 loss sweep | 实验量爆炸，先只看 normal FP loss |
| d2 / d1 / x0 skip gate | 当前收益不确定，且浅层 gate 可能压掉细裂缝 |
| broken 样本训练 | 当前主数据集已排除 broken，mask 可靠性也不一致 |

---

## 11. 评价指标

主表至少报告：

| 指标 | 说明 |
|---|---|
| Dice | 主分割指标 |
| IoU | 主分割指标 |
| Precision | 看 false positive |
| Recall | 看漏检 |
| F1 | precision / recall 平衡 |
| Normal FPR | normal 图是否被误报为 crack |
| Normal FP pixels / area | normal 图误报面积 |

建议补充：

| 指标 | 说明 |
|---|---|
| Boundary F1 | 裂缝边界是否准确 |
| Component F1 | 连通裂缝区域是否检测正确 |
| Image-level crack recall | 有裂缝图是否被检测到 |

分组评估：

```text
overall test
by label: crack / normal
by device or source: camera / phone / dphone
by image size
```

如果 train/val/test 存在同一 `video_name` 分布重叠，不需要重新切数据，但需要在实验记录里说明：数据已人工筛掉相邻帧，当前采用 fixed split。

---

## 12. 结果表模板

### 12.1 主模型表

| ID | model | Dice ↑ | IoU ↑ | Precision ↑ | Recall ↑ | Normal FPR ↓ | Boundary F1 ↑ | comment |
|---|---|---:|---:|---:|---:|---:|---:|---|
| M0 | baseline |  |  |  |  |  |  |  |
| M1 | bottleneck attention |  |  |  |  |  |  |  |
| M2 | skip d4 |  |  |  |  |  |  |  |
| M3 | skip d3 |  |  |  |  |  |  |  |
| M4 | skip d4+d3 |  |  |  |  |  |  |  |
| M5 | self-attention d4+d3 |  |  |  |  |  |  |  |
| M6 | prototype cross-attention |  |  |  |  |  |  |  |

正式写法：

```text
mean ± std over 3 seeds
```

### 12.2 baseline 训练机制消融表

| ID | setting | Dice ↑ | Recall ↑ | Normal FPR ↓ | conclusion |
|---|---|---:|---:|---:|---|
| T0 | full baseline |  |  |  |  |
| T1 | no Stage1 |  |  |  |  |
| T2 | no hard normal replay |  |  |  |  |
| T3 | normal FP loss |  |  |  |  |

### 12.3 后处理消融表

| ID | threshold | min_area | Dice ↑ | Precision ↑ | Recall ↑ | Normal FPR ↓ |
|---|---|---:|---:|---:|---:|---:|
| P0 | 0.50 | 0 |  |  |  |  |
| P1 | val selected | 0 |  |  |  |  |
| P2 | val selected | val selected |  |  |  |  |

### 12.4 source-wise test 表

| model | source/device | Dice ↑ | Precision ↑ | Recall ↑ | Normal FPR ↓ |
|---|---|---:|---:|---:|---:|
| M0 | camera |  |  |  |  |
| M0 | phone |  |  |  |  |
| M0 | dphone |  |  |  |  |
| M4 | camera |  |  |  |  |
| M4 | phone |  |  |  |  |
| M4 | dphone |  |  |  |  |
| M5 | camera |  |  |  |  |
| M5 | phone |  |  |  |  |
| M5 | dphone |  |  |  |  |

---

## 13. 实验命名

统一命名：

```text
811_m0_baseline_s20260515
811_m0_baseline_s20260516
811_m0_baseline_s20260517

811_m1_tbn_d1_s20260515
811_m2_skip_d4_s20260515
811_m3_skip_d3_s20260515
811_m4_skip_d4d3_s20260515
811_m5_selfattn_d4d3_s20260515
811_m6_hnproto_s20260515

811_t1_no_stage1_s20260515
811_t2_no_hard_normal_s20260515
811_t3_normal_fp_loss_s20260515
```

每个 run 至少保存：

```text
config.yaml
best_stage1.pt
best_stage2.pt
val_metrics.json
holdout/holdout_metrics.json
holdout/holdout_per_image.csv
selected_postprocess.json
train_log.csv
prediction_visualizations/
```

对于共享 Stage1 的 attention run，可以只在对应 seed 的 baseline 目录保存 `best_stage1.pt`，其他 run 在 config 里记录加载路径。

---

## 14. 最小实验集

时间紧时，最少跑：

```text
1. M0 baseline, 3 seeds
2. M1 bottleneck attention, 3 seeds
3. M4 skip attention d4+d3, 3 seeds
4. M5 self-attention d4+d3, 3 seeds
5. M6 prototype cross-attention, 1 seed
6. T1 no Stage1, 1 seed
7. T2 no hard normal replay, 1 seed
8. P0 / P1 / P2 后处理对比
```

总训练量：

```text
4 models × 3 seeds + M6 × 1 seed + 2 ablations × 1 seed = 15 Stage2 runs
```

这个版本可以支撑主要结论：

```text
baseline 是否可靠
two-stage 是否必要
hard normal replay 是否必要
bottleneck attention 是否有效
skip attention gate 是否有效
self-attention 是否比 attention gate 更适合
prototype cross-attention 是否值得深入
后处理对最终结果影响多大
```

---

## 15. 推荐完整实验集

算力允许时，推荐完整跑：

```text
M0 baseline × 3 seeds
M1 bottleneck attention × 3 seeds
M2 skip d4 × 3 seeds
M3 skip d3 × 3 seeds
M4 skip d4+d3 × 3 seeds
M5 self-attention d4+d3 × 3 seeds
M6 prototype cross-attention × 1-3 seeds
T1 no Stage1 × 1 seed
T2 no hard normal replay × 1 seed
T3 normal FP loss × 1-2 seeds
P0 / P1 / P2 postprocess evaluation
overall / source-wise / normal-FP test evaluation
```

总训练量：

```text
18 main Stage2 runs + M6 1-3 runs + 3-4 baseline ablation runs = 22-25 Stage2 runs
```

可选再加：

```text
tbn + skipgate combination × 1-3 seeds
full M6 prototype cross-attention × 3 seeds
```

---

## 16. 结果解释规则

### 16.1 bottleneck attention

如果 M1 相比 M0：

```text
Dice / IoU 提升
Recall 提升或保持
Normal FPR 没有明显变差
```

可以说 bottleneck attention 对全局上下文建模有帮助。

如果 Recall 提升但 Normal FPR 也升高，说明它可能更激进，需要结合 threshold 或 normal FP loss。

### 16.2 skip attention gate

如果 M2/M3/M4 相比 M0：

```text
Precision 提升
Normal FPR 下降
Recall 基本不下降
```

可以说 skip gate 有助于过滤金属纹理、反光、背景划痕导致的误检。

如果 Normal FPR 降低但 Recall 明显下降，说明 gate 太强，可能把细裂缝也压掉了。

### 16.3 self-attention

如果 M5 相比 M0：

```text
Dice / IoU 提升
Recall 提升
Boundary F1 或细裂缝可视化更好
```

可以说 decoder self-attention 有助于建模长程裂缝结构。

如果 M5 提升不如 M4，但参数和训练开销更大，则说明当前数据上显式 skip filtering 比 decoder token mixing 更实用。

### 16.4 prototype cross-attention

如果 M6 相比 M1：

```text
Normal FPR 更低
hard normal 上 FP 更少
crack recall 没有明显下降
```

可以说 prototype cross-attention 利用了正/负原型先验，对区分裂缝和金属干扰纹理有帮助。

如果 M6 不稳定或只在单 seed 上提升，则不要把它作为主结论，可以放进附表作为探索性实验。

### 16.5 d4 vs d3

| 结果 | 解释 |
|---|---|
| d4 最好 | 高层语义 skip 更需要过滤，浅层保留细节更重要 |
| d3 最好 | 中层纹理/结构特征过滤最有效 |
| d4+d3 最好 | 多尺度 skip filtering 有互补效果 |
| 都不如 baseline | 当前 skip gate 不适合，或数据量不足导致 gate 学不稳定 |

---

## 17. 论文/报告主线

英文主线：

```text
1. We establish a ResNet34-UNet baseline on a fixed 8/1/1 crack/normal split.
2. We validate the training pipeline by ablating Stage1 pretraining and hard normal replay.
3. We evaluate bottleneck attention for global crack representation.
4. We evaluate skip attention gates for suppressing false positives from metallic textures.
5. We evaluate decoder self-attention as an alternative to attention gates.
6. We optionally evaluate prototype cross-attention using positive and hard-negative prototypes.
7. We use validation-selected threshold and min_area for final test evaluation.
8. We report overall, source-wise, and normal false-positive metrics.
```

中文主线：

```text
先固定 8/1/1 split 和 canonical baseline；
再验证 Stage1 / hard normal replay 是否必要；
然后比较 bottleneck attention、skip attention gate、self-attention；
可选补充 prototype cross-attention；
最后用 val 选择后处理参数，在 test 上一次性评估。
```

---

## 18. 最终执行清单

按顺序执行：

```text
[ ] 用 dataset_crack_normal_unet_811 重新生成 manifests/samples.csv
[ ] 检查 manifest split / label / device / mask path
[ ] 确认 active config 无 fold / cv_fold / n_folds
[ ] 检查 Stage2-only 训练逻辑，支撑 T1
[ ] 检查 holdout quantitative evaluator，支撑 test metrics 和 source-wise 表
[ ] 跑 M0-sanity，确认 mask 对齐和 evaluation 正常
[ ] 跑 T1/T2/T3 baseline 训练机制消融
[ ] 锁定正式 baseline 设置
[ ] 跑 P0/P1/P2 后处理评估，锁定统一 postprocess protocol
[ ] 按 seed 训练共享 Stage1 checkpoint
[ ] 跑 M0/M1/M4/M5 正式 3 seeds
[ ] 先跑 M2/M3 单 seed，必要时补齐 3 seeds
[ ] 构建 prototype bank，跑 M6 单 seed，必要时补齐 3 seeds
[ ] 汇总 overall test metrics
[ ] 汇总 source-wise / image-size-wise / normal false-positive metrics
[ ] 选 5-10 张典型可视化：TP、FN、FP、normal false positive、细裂缝
[ ] 有时间再跑 tbn+skipgate 或补齐 M6 3 seeds
```

---

## 19. 一句话版本

最终主实验是：

```text
M0: ResNet34-UNet baseline
M1: baseline + bottleneck attention
M2: baseline + skip attention d4
M3: baseline + skip attention d3
M4: baseline + skip attention d4+d3
M5: baseline + decoder self-attention d4+d3
M6: baseline + prototype cross-attention
```

再加 baseline 上的关键消融：

```text
T1: no Stage1 pretraining
T2: no hard normal replay
T3: normal FP loss
```

所有模型统一：

```text
fixed 8/1/1 split
canonical baseline 派生配置
val 选择 threshold / min_area
test 只评最终结果
主模型跑 3 seeds
不跑 4-fold
```
