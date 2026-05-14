# Attention Comparison Runner

This folder owns cross-experiment execution and analysis for the four attention experiments.

Run all four experiments sequentially, with folds parallelized across four GPUs:

```bash
bash experiments/attention_20260511/comparison/run_all_4gpu.sh
```

Run only the fairness check:

```bash
python experiments/attention_20260511/comparison/check_fairness.py
```

Summarize completed runs:

```bash
python experiments/attention_20260511/comparison/summarize_attention_experiments.py \
  --experiment-root experiments/attention_20260511 \
  --output-dir experiments/attention_20260511/comparison/results
```

The comparison runner deliberately does not tune on holdout. It uses pooled OOF to select the global threshold/min_area, then applies that frozen setting to holdout.
