# Active Dataset Description

Active dataset:

```text
/Users/yangyi/Desktop/masterthesis/dataset_crack_normal_unet_811
```

This dataset replaces the old cross-validation setup. It contains only `crack`
and `normal` samples and already provides a fixed `train`/`val`/`test` split, so
the project now uses one canonical train/validation/test run.

## Split Protocol

| split | crack | normal | total | role |
| --- | ---: | ---: | ---: | --- |
| `train` | 371 | 274 | 645 | Stage1/Stage2 training |
| `val` | 47 | 34 | 81 | validation and threshold selection |
| `test` | 45 | 35 | 80 | final holdout inference/evaluation |
| total | 463 | 343 | 806 | |

The split is 8/1/1 and was stratified by source dataset plus label. `broken`
samples are excluded.

## Layout

```text
dataset_crack_normal_unet_811/
  train/
    images/crack/
    images/normal/
    masks/crack/
    masks/normal/
    annotations/crack/
  val/
    images/crack/
    images/normal/
    masks/crack/
    masks/normal/
    annotations/crack/
  test/
    images/crack/
    images/normal/
    masks/crack/
    masks/normal/
    annotations/crack/
  manifest.csv
  summary.json
  DATASET_DESCRIPTION.md
```

Each image has one binary mask. Crack masks are generated from LabelMe polygons;
normal masks are all black.

## Project Mapping

`scripts/prepare_samples.py` reads `manifest.csv` and writes
`manifests/samples.csv` with the same split protocol:

- `split=train`: used for Stage1/Stage2 training.
- `split=val`: used for validation.
- `split=test`: used by holdout inference.

The old `trainval + cv_fold` protocol belongs to the frozen historical archive,
not the active baseline code path.
