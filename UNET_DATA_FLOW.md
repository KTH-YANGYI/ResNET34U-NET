# Active UNet Data Flow

The active project no longer uses four-fold cross-validation. The current data
contract is the fixed split dataset:

```text
/Users/yangyi/Desktop/masterthesis/dataset_crack_normal_unet_811
```

## Flow

```text
dataset manifest.csv
  -> scripts/prepare_samples.py
  -> manifests/samples.csv
  -> scripts/build_patch_index.py
  -> manifests/stage1_train_index.csv
  -> manifests/stage1_val_index.csv
  -> scripts/train_stage1.py
  -> outputs/experiments/canonical_baseline/stage1/best_stage1.pt
  -> scripts/train_stage2.py
  -> outputs/experiments/canonical_baseline/stage2/best_stage2.pt
  -> scripts/evaluate_val.py
  -> outputs/experiments/canonical_baseline/stage2/val_metrics.json
  -> scripts/infer_holdout.py
  -> outputs/experiments/canonical_baseline/stage2/holdout/
```

## Splits

- `train`: Stage1 patch training and Stage2 full-image training.
- `val`: validation and post-processing threshold/min-area selection.
- `test`: final holdout inference/evaluation.

There is no `cv_fold`, OOF pooling, or fold ensemble in the active code path.
Historical cross-validation notes are kept only inside the frozen old-experiment
archive.
