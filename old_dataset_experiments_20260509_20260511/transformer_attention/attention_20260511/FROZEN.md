# Frozen Four-Experiment Archive

Frozen date: 2026-05-14

This directory is a read-only historical archive for the completed old-dataset
attention comparison:

1. `00_baseline_resnet34_unet_pw12`
2. `01_tbn_d1`
3. `02_tbn_d1_hnproto`
4. `03_skipgate_d4d3`

The configs, logs, fold outputs, OOF search results, holdout outputs, and
comparison tables here should not be edited for new work. Use them only for
reproduction, result lookup, and thesis reporting.

The original Codex execution plan for this transformer/attention work is stored
at:

```text
old_dataset_experiments_20260509_20260511/transformer_attention/plans/CODEX_TRANSFORMER_EXPERIMENT_PLAN.md
```

Current and future experiments must derive from:

```text
configs/canonical_baseline.yaml
```
