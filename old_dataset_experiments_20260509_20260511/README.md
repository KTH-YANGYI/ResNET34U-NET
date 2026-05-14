# Old-Dataset U-Net Experiment Archive

这个文件夹专门保存旧数据集上完成的 U-Net 消融实验和 transformer/attention 实验。
原始实验目录没有移动或删除；这里是为了后续新数据集重跑、论文整理和仓库清理做的一份集中归档。

## Scope

这些结果都来自旧数据集，不应直接和后续新数据集实验混合比较。

包含两组实验：

1. `ablations/`
   2026-05-09 的 U-Net two-stage 消融实验，包括 `pos_weight`、`normal_fp_loss_weight`、deep supervision、boundary auxiliary loss 等。

2. `transformer_attention/`
   2026-05-11 的 transformer / attention 结构实验，包括 bottleneck transformer、hard-normal prototype attention、skip attention gate 等。

相关执行计划：

- `plans/REPO_EXPERIMENT_CONFIG_AND_CLEANUP_PLAN.md`
  正式 baseline、四实验配置和仓库清理执行计划。
- `transformer_attention/plans/CODEX_TRANSFORMER_EXPERIMENT_PLAN.md`
  Transformer/attention 三个创新点的 Codex 执行计划。

## Ablation Experiments

位置：

- `ablations/pos_weight_ablation_20260509_review/`
- `ablations/normal_fp_loss_ablation_20260509_review/`
- `ablations/third_batch_segmentation_20260509_review/`
- `ablations/configs/`
- `ablations/scripts/`

主要结论：

- `pos_weight=12` 是旧数据集上最稳的 Stage2 设置。
- `pos_weight=6/8/12` 中，`pos_weight=12` 保持最高 recall，并且 normal false positive 最低。
- `normal_fp_loss_weight=0.03/0.05` 没有提升 baseline，反而降低 Dice 或增加 normal false positive。
- Deep supervision 略微提升 IoU，但降低 recall，不适合作为最终默认配置。
- Boundary auxiliary loss 保持接近 baseline 的 recall，但 normal false positive 增加，也不是最佳默认选择。

旧数据集上推荐的非-transformer baseline：

- Stage2 `pos_weight=12`
- 固定 threshold/min_area checkpoint 策略
- hard-normal replay 带采样上限
- OOF global post-processing search

## Transformer / Attention Experiments

位置：

- `transformer_attention/attention_20260511/`
- `transformer_attention/configs/transformer_a40_20260510/`
- `transformer_attention/scripts/`
- `transformer_attention/plans/CODEX_TRANSFORMER_EXPERIMENT_PLAN.md`

冻结状态：

- 这四个实验是旧数据集上的历史结果，只用于复现、论文对照和结果追溯。
- 不再从这些目录派生新实验；新实验必须从仓库根目录的 `configs/canonical_baseline.yaml` 派生。
- 不要修改这里的 `config.yaml`、`run.sh`、`logs/` 或 `results/`。如果需要新的对照，复制 canonical baseline 到新的实验配置并重跑。

对照和变体：

1. `00_baseline_resnet34_unet_pw12`
   ResNet34-U-Net baseline。

2. `01_tbn_d1`
   Bottleneck Transformer，1 层 Transformer encoder，8 heads。

3. `02_tbn_d1_hnproto`
   Bottleneck Transformer + hard-normal prototype cross-attention。

4. `03_skipgate_d4d3`
   Decoder skip attention gate，作用于 `d4` 和 `d3` skip features。严格来说它是 attention gate，不是 transformer bottleneck。

旧数据集上的主要结论：

- `01_tbn_d1` 是最稳的 transformer 改法：Dice/IoU 小幅提升，recall 基本不掉。
- `02_tbn_d1_hnproto` 的 OOF Dice/IoU 最高，但 recall 略降，适合作为论文里的 trade-off 实验。
- `03_skipgate_d4d3` 表现变差，不建议作为主线。

关键结果文件：

- `transformer_attention/attention_20260511/comparison/results/ATTENTION_EXPERIMENT_SUMMARY.md`
- `transformer_attention/attention_20260511/comparison/results/FAIRNESS_CHECK.md`
- `transformer_attention/attention_20260511/comparison/results/attention_oof_metrics.csv`
- `transformer_attention/attention_20260511/comparison/results/attention_oof_deltas.csv`
- `transformer_attention/attention_20260511/comparison/results/attention_holdout_metrics.csv`

## Notes For New Dataset Work

这些实验可以作为方法设计和论文叙述的依据，但新数据集需要重新跑：

- 新的 train/validation/test split
- 新的 Stage1 patch index
- 新的 Stage2 fold or validation protocol
- 新的 OOF 或 validation threshold/min_area selection
- 新的 ablation and transformer comparison if paper claims quantitative improvement

如果后续只保留核心代码，可以保留这个 archive 作为旧数据集实验证据；如果要让 GitHub 更干净，也可以把该文件夹作为唯一旧实验归档入口。

## Legacy Root Configs And Scripts

位置：

- `legacy_root_configs/`
- `legacy_root_scripts/`

这些文件是从仓库根目录移出的旧活入口备份，包括 P0/A40 消融配置和旧 runner。它们不是当前项目入口，只用于追溯旧实验命令。
