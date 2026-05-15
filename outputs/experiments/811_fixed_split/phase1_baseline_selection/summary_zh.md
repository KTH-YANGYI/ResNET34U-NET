# 811 fixed split 实验结果汇总

- 生成时间：2026-05-15 16:10:21
- 实验目录：`outputs/experiments/811_fixed_split/phase1_baseline_selection`
- scope：`phase1`
- baseline profile：`full`

## 当前结论

当前 validation 排名第一的是 `811_m0_baseline_s20260515`（M0 full baseline，seed=20260515），val stage2_score=0.316180，val Dice=0.527944，val normal FPR=0.205882。
正式结论仍以所有计划内变体完成后的同表比较为准。

## 指标表

| 实验 | 状态 | 变体 | seed | val score | val Dice | val IoU | val normal FPR | val 后处理 | holdout Dice | holdout IoU | holdout normal FPR |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 811_m0_baseline_s20260515 | 完成 | M0 full baseline | 20260515 | 0.316180 | 0.527944 | 0.422235 | 0.205882 | thr=0.950000, area=48 | 0.482640 | 0.385068 | 0.000000 |
| 811_t3_normal_fp_loss_s20260515 | 完成 | T3 normal FP loss | 20260515 | 0.305737 | 0.399854 | 0.302128 | 0.147059 | thr=0.950000, area=48 | 0.401525 | 0.300942 | 0.000000 |
| 811_t2_no_hard_normal_s20260515 | 完成 | T2 no hard normal mining | 20260515 | 0.231507 | 0.384448 | 0.294797 | 0.176471 | thr=0.950000, area=48 | 0.385045 | 0.302615 | 0.000000 |
| 811_t1_no_stage1_s20260515 | 完成 | T1 no Stage1 | 20260515 | -0.960601 | 0.368811 | 0.269191 | 0.764706 | thr=0.950000, area=48 | 0.411856 | 0.304668 | 0.000000 |

## 目录说明

每个实验都有独立目录：`<run>/stage2/` 保存 checkpoint、history、resolved config、validation 指标；`<run>/stage2/holdout/` 保存 holdout/test 指标和逐图结果。
权重文件不建议推到 GitHub；用于论文和排查的 CSV/JSON/log 可以保留。
