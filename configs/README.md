# Config Policy

`canonical_baseline.yaml` is the only active baseline entry for the current
project state.

Historical ablation and transformer/attention configs are frozen under:

```text
old_dataset_experiments_20260509_20260511/
```

New experiments should derive from the canonical baseline with `extends` and
only override the method-specific fields:

```yaml
extends: ../canonical_baseline.yaml
name: my_new_experiment

stage2:
  model_name: my_new_experiment
  model_variant: tbn_d1
  transformer_bottleneck_layers: 1
  stage1_load_strict: false
  save_dir_template: outputs/experiments/my_new_experiment/stage2/fold{fold}
  global_postprocess_path: outputs/experiments/my_new_experiment/stage2/oof_global_postprocess.json
```

Model architecture is selected through `stage2.model_variant`; the baseline
config should not accumulate experiment-specific `*_enable` switches. Frozen
multi-run comparisons belong under `experiment_sets/`.

Formal baseline runs keep `allow_pretrained_fallback: false` and
`auto_evaluate_strict: true` so missing weights or failed validation exports
surface immediately. Patch-index outputs should stay behind the configured
`train_index_path_template` and `val_index_path_template`.

Keep the canonical file unchanged unless the thesis baseline itself is being
redefined.
