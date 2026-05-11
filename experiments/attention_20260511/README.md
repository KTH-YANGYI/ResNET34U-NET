# Attention Architecture Experiments - 2026-05-11

This folder is the formal, pre-run workspace for the four Stage2 attention/architecture comparisons.

The layout is intentionally experiment-centric:

```text
experiments/attention_20260511/
  00_baseline_resnet34_unet_pw12/
    config.yaml
    run.sh
    logs/
    results/
    README.md
  01_tbn_d1/
    config.yaml
    run.sh
    logs/
    results/
    README.md
  02_tbn_d1_hnproto/
    config.yaml
    run.sh
    logs/
    results/
    README.md
  03_skipgate_d4d3/
    config.yaml
    run.sh
    logs/
    results/
    README.md
  comparison/
    run_all_4gpu.sh
    run_single_experiment.sh
    check_fairness.py
    summarize_attention_experiments.py
    paired_oof_significance.py
    alvis_attention_4a100.sbatch
```

The four experiment folders are parallel and self-contained. Each folder owns its config, local run wrapper, logs, and results. The `comparison` folder owns cross-experiment orchestration, fairness checks, final tables, and paired analyses.

## Four Experiments

| id | folder | purpose |
| --- | --- | --- |
| 00 | `00_baseline_resnet34_unet_pw12` | Best stable ResNet34-U-Net Stage2 baseline with `pos_weight=12` |
| 01 | `01_tbn_d1` | Add one transformer bottleneck layer at the lowest-resolution U-Net feature map |
| 02 | `02_tbn_d1_hnproto` | Add hard-normal prototype cross-attention on top of the transformer bottleneck |
| 03 | `03_skipgate_d4d3` | Add decoder skip attention gates at `d4` and `d3` |

## Fairness Rules

All four experiments must keep the following fixed:

- Same sample manifest: `manifests/samples.csv`
- Same folds: `0 1 2 3`
- Same Stage1 checkpoints: `outputs/experiments/p0_a40_20260508/stage1/fold{fold}/best_stage1.pt`
- Same image size: `640`
- Same Stage2 batch size: `48`
- Same Stage2 epochs: `50`
- Same optimizer and learning rates: encoder `2e-5`, decoder `1e-4`, weight decay `1e-4`
- Same loss weights: BCE `0.5`, Dice `0.5`, `pos_weight=12`, `normal_fp_loss_weight=0`
- Same augmentation policy
- Same hard-normal replay policy and cap
- Same checkpoint-selection rule and OOF post-processing search grid
- Same holdout rule: holdout uses frozen OOF-selected threshold/min_area and must not be tuned

Only the architecture switches are allowed to differ.

## Run

From the repository root:

```bash
bash experiments/attention_20260511/comparison/run_all_4gpu.sh
```

Or on Alvis:

```bash
sbatch experiments/attention_20260511/comparison/alvis_attention_4a100.sbatch
```

To run a single experiment:

```bash
bash experiments/attention_20260511/01_tbn_d1/run.sh
```

Run artifacts stay inside each experiment's own `logs/` and `results/` folders.
