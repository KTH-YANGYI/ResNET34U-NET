# Config Policy

`canonical_baseline.yaml` is the only active baseline entry for the current
project state.

Historical ablation and transformer/attention configs are frozen under:

```text
old_dataset_experiments_20260509_20260511/
```

New experiments should derive from the canonical baseline with `extends` and
only override the method-specific fields. The active dataset uses an explicit
`train`/`val`/`test` split; there is no cross-validation layer in active configs:

```yaml
extends: ../canonical_baseline.yaml
name: my_new_experiment

stage2:
  model_name: my_new_experiment
  model_variant: tbn_d1
  transformer_bottleneck_layers: 1
  stage1_load_strict: false
  save_dir: outputs/experiments/my_new_experiment/stage2
```

Current planned local experiment configs:

- `configs/experiments/self_attention_d4d3.yaml`: decoder `d4/d3`
  residual self-attention (`model_variant: selfattn_d4d3`).
- `configs/experiments/prototype_cross_attention.yaml`: bottleneck transformer
  plus positive/hard-negative prototype cross-attention
  (`model_variant: tbn_d1_hnproto`).

Experiment configs normally reuse the canonical Stage1 checkpoint. Run them
with `scripts/train_pipeline.sh --stage2-only`, or explicitly set independent
Stage1 paths before allowing a full Stage1 run.

Model architecture is selected through `stage2.model_variant`; the baseline
config should not accumulate experiment-specific `*_enable` switches. Frozen
multi-run comparisons belong under `experiment_sets/`.

Formal baseline runs keep `allow_pretrained_fallback: false` and
`auto_evaluate_strict: true` so missing weights or failed validation exports
surface immediately. Patch-index outputs should stay behind the configured
`train_index_path` and `val_index_path`.

Keep the canonical file unchanged unless the thesis baseline itself is being
redefined.
