# Original Model Data Flow

The active baseline is a two-stage ResNet34-U-Net pipeline on the fixed
`train`/`val`/`test` dataset split.

## Stage1

Stage1 trains the same ResNet34-U-Net architecture on mined image patches:

```text
manifests/stage1_train_index.csv
manifests/stage1_val_index.csv
  -> scripts/train_stage1.py
  -> outputs/experiments/canonical_baseline/stage1/best_stage1.pt
```

Its purpose is local crack-feature pretraining: the patch index increases the
positive-pixel density seen by the model before full-image training.

## Stage2

Stage2 initializes from the Stage1 checkpoint and fine-tunes on full images:

```text
manifests/samples.csv
  split=train -> training
  split=val   -> validation and post-process search
  split=test  -> final holdout inference
```

Outputs are written under:

```text
outputs/experiments/canonical_baseline/stage2/
```

The canonical architecture is selected by:

```yaml
stage2:
  model_variant: resnet34_unet_baseline
```

New architecture experiments should extend `configs/canonical_baseline.yaml`
and override only `stage2.model_variant` plus the variant-specific fields.
The frozen 2026-05-11 attention comparison remains documented under
`experiment_sets/attention_20260511.yaml`.
