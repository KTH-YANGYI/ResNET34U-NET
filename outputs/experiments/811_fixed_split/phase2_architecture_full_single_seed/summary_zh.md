# 811 fixed split 实验结果汇总

- 生成时间：2026-05-16 07:06:03
- 实验目录：`outputs/experiments/811_fixed_split/phase2_architecture_full_single_seed`
- scope：`phase2_single`
- baseline profile：`full`

## 当前结论

当前 validation 排名第一的是 `811_m4_skip_d4d3_s20260515`（M4 skip attention gate d4+d3，seed=20260515），val stage2_score=0.548863，val Dice=0.642980，val normal FPR=0.147059。
正式结论仍以所有计划内变体完成后的同表比较为准。

## 指标表

| 实验 | 状态 | 变体 | seed | val score | val Dice | val IoU | val normal FPR | val 后处理 | holdout Dice | holdout IoU | holdout normal FPR |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 811_m4_skip_d4d3_s20260515 | 完成 | M4 skip attention gate d4+d3 | 20260515 | 0.548863 | 0.642980 | 0.524919 | 0.147059 | thr=0.900000, area=0 | 0.616708 | 0.504171 | 0.000000 |
| 811_m5_selfattn_d4d3_s20260515 | 完成 | M5 decoder self-attention d4+d3 | 20260515 | 0.547246 | 0.641364 | 0.525623 | 0.147059 | thr=0.950000, area=0 | 0.595558 | 0.485301 | 0.000000 |
| 811_m2_skip_d4_s20260515 | 完成 | M2 skip attention gate d4 | 20260515 | 0.537255 | 0.690196 | 0.558024 | 0.176471 | thr=0.650000, area=48 | 0.639879 | 0.520736 | 0.000000 |
| 811_m1_tbn_d1_s20260515 | 完成 | M1 bottleneck transformer | 20260515 | 0.507456 | 0.601574 | 0.484546 | 0.147059 | thr=0.800000, area=48 | 0.583792 | 0.476202 | 0.000000 |
| 811_m6_hnproto_s20260515 | 完成 | M6 prototype cross-attention | 20260515 | 0.438950 | 0.533067 | 0.419990 | 0.147059 | thr=0.950000, area=48 | 0.514067 | 0.406107 | 0.000000 |
| 811_m3_skip_d3_s20260515 | 完成 | M3 skip attention gate d3 | 20260515 | 0.432074 | 0.467368 | 0.373783 | 0.117647 | thr=0.950000, area=0 | 0.453428 | 0.366761 | 0.000000 |
| 811_m0_baseline_s20260515 | 完成 | M0 full baseline | 20260515 | 0.416140 | 0.569081 | 0.457011 | 0.176471 | thr=0.950000, area=48 | 0.544464 | 0.438613 | 0.000000 |

## 目录说明

每个实验都有独立目录：`<run>/stage2/` 保存 checkpoint、history、resolved config、validation 指标；`<run>/stage2/holdout/` 保存 holdout/test 指标和逐图结果。
权重文件不建议推到 GitHub；用于论文和排查的 CSV/JSON/log 可以保留。
