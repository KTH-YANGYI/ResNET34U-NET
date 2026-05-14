# UNET Two-Stage

Two-stage defect segmentation pipeline:

1. Stage1 trains a patch-level model on cropped positive and negative patches.
2. Stage2 loads the best Stage1 checkpoint and trains a full-image segmentation model.
3. Validation export and holdout inference reuse the post-processing thresholds selected on the validation set.

## Active Config Policy

The current live entry point is:

```text
configs/canonical_baseline.yaml
```

This one file contains both the `stage1` and `stage2` baseline sections. All
current commands point to it, and future experiment configs should use
`extends: ../canonical_baseline.yaml` with only method-specific overrides.
Stage2 architecture is selected with `stage2.model_variant`; the canonical
baseline uses `resnet34_unet_baseline`.

The completed 2026-05-11 four-experiment attention comparison is frozen under
`old_dataset_experiments_20260509_20260511/` and is also summarized as an
experiment set manifest at `experiment_sets/attention_20260511.yaml`. It is no
longer an active training entry.

## Environment

Recommended setup:

- Python `3.10+`
- CUDA GPU

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use `conda`:

```bash
conda create -n unet_two_stage python=3.10 -y
conda activate unet_two_stage
pip install -r requirements.txt
```

## Project Assumptions

- The project root contains `configs/`, `scripts/`, and `src/`.
- `UNET_two_stage/` and `dataset0505_crop640_roi/` are sibling directories under the same parent directory.
- Generated paths under `manifests/`, `generated_masks/`, and `outputs/` are resolved relative to the project root.
- Default configs use 4 folds: `0`, `1`, `2`, `3`.

## Dataset Layout

`scripts/prepare_samples.py` expects the ROI dataset under `../dataset0505_crop640_roi/` by default:

```text
parent_dir/
|- UNET_two_stage/
`- dataset0505_crop640_roi/

dataset0505_crop640_roi/
|- camera/
|  |- crack/
|  |- normal/
|  `- broken/
`- phone/
   |- crack/
   |- normal/
   `- broken/
```

Notes:

- `crack` images must have same-stem LabelMe JSON files; masks are generated under `generated_masks/`.
- `normal` images are used as negative samples.
- `broken` images are unlabeled holdout samples by default.
- By default, `prepare_samples.py` also reserves 20% of labeled `crack` and `normal` images as inference-only holdout with a fixed seed.
- Cross-validation is image-level random splitting; video mapping files are not used.

### Dataset Roles And Counts

For the current `dataset0505_crop640_roi` snapshot:

| subset | crack | normal | broken | total |
| --- | ---: | ---: | ---: | ---: |
| camera | 157 | 38 | 44 | 239 |
| phone | 106 | 36 | 63 | 205 |
| total | 263 | 74 | 107 | 444 |

The classes are used as follows:

- `crack`: labeled defect images. LabelMe JSON annotations are converted to binary masks under `generated_masks/`.
- `normal`: negative images used for training and validation.
- `broken`: unlabeled images used only for holdout inference. They are never used by Stage1, Stage2, or CV validation.

With the default split (`--test-ratio 0.20 --test-seed 2026`), `prepare_samples.py` produces:

| split | crack | normal | broken | total | role |
| --- | ---: | ---: | ---: | ---: | --- |
| `trainval` | 211 | 59 | 0 | 270 | Stage1/Stage2 training and CV validation |
| `holdout` | 52 | 15 | 107 | 174 | inference only |

Each fold trains on about 200 original images and validates on 66-70 original images. Stage1 expands those images into patch indexes; the Stage1 `batch_size` is a patch batch size, while the Stage2 `batch_size` is a full-image batch size.

## Pipeline

### 1. Generate manifests

```bash
python scripts/prepare_samples.py
```

The default split keeps 20% of `crack` and `normal` out of all training and validation folds:

```bash
python scripts/prepare_samples.py --test-ratio 0.20 --test-seed 2026
```

Use `--test-ratio 0` only when you want every labeled/normal sample available for cross-validation.

If the dataset is somewhere else, pass it explicitly:

```bash
python scripts/prepare_samples.py --dataset-root /path/to/dataset0505_crop640_roi
```

This step scans the dataset and writes:

- `manifests/samples.csv`
- `manifests/samples_summary.json`
- `generated_masks/{device}/crack/*.png`

Rows with `split=trainval` are used by Stage1/Stage2. Rows with `split=holdout` are never used for training and are consumed by `scripts/infer_holdout_ensemble.py`.

Run this step again on the training server. `manifests/samples.csv` stores absolute image and mask paths for the current machine, so a local copy should not be reused after moving to a server.

### 2. Build Stage1 patch index

```bash
python scripts/build_patch_index.py --config configs/canonical_baseline.yaml
```

Run this after `prepare_samples.py` on the server so patch indexes also point to the server paths.

This step writes:

- `manifests/stage1_fold{fold}_train_index.csv`
- `manifests/stage1_fold{fold}_val_index.csv`
- `manifests/stage1_patch_summary.json`

### 3. Train Stage1

```bash
python scripts/train_stage1.py --config configs/canonical_baseline.yaml --fold 0
```

Default Stage1 save directory from `configs/canonical_baseline.yaml`:

```text
outputs/experiments/canonical_baseline/stage1/fold{fold}/
```

### 4. Train Stage2

```bash
python scripts/train_stage2.py --config configs/canonical_baseline.yaml --fold 0
```

Stage2 uses:

- `manifests/samples.csv`
- `outputs/experiments/canonical_baseline/stage1/fold{fold}/best_stage1.pt`

Default Stage2 save directory from `configs/canonical_baseline.yaml`:

```text
outputs/experiments/canonical_baseline/stage2/fold{fold}/
```

By default, `train_stage2.py` also runs validation export automatically after training because `auto_evaluate_after_train: true` in `configs/canonical_baseline.yaml`.

Stage2 hard normal replay keeps the normal-image budget fixed. With `stage2_hard_normal_ratio: 0.40`, about 40% of the normal samples in an epoch come from the mined hard-normal pool once that pool exists, and the rest stay randomly sampled normals.

### Search a global OOF post-process setting

After all folds have Stage2 checkpoints, pool the out-of-fold validation predictions and select one global threshold/min-area pair:

```bash
python scripts/search_oof_postprocess.py --config configs/canonical_baseline.yaml --folds 0,1,2,3
```

This writes:

- `outputs/experiments/canonical_baseline/stage2/oof_global_postprocess.json`
- `outputs/experiments/canonical_baseline/stage2/oof_global_postprocess_search.csv`
- `outputs/experiments/canonical_baseline/stage2/oof_per_image.csv`

For final fold-ensemble holdout inference, use `scripts/infer_holdout_ensemble.py` after `global_postprocess_path` exists.

### Run All Folds

To reproduce all folds, either run the fold commands explicitly or use the
canonical wrapper:

```bash
bash scripts/train_all_folds.sh --gpus 0,1,2,3
```

The explicit fold commands are:

```bash
python scripts/train_stage1.py --config configs/canonical_baseline.yaml --fold 0
python scripts/train_stage1.py --config configs/canonical_baseline.yaml --fold 1
python scripts/train_stage1.py --config configs/canonical_baseline.yaml --fold 2
python scripts/train_stage1.py --config configs/canonical_baseline.yaml --fold 3

python scripts/train_stage2.py --config configs/canonical_baseline.yaml --fold 0
python scripts/train_stage2.py --config configs/canonical_baseline.yaml --fold 1
python scripts/train_stage2.py --config configs/canonical_baseline.yaml --fold 2
python scripts/train_stage2.py --config configs/canonical_baseline.yaml --fold 3

python scripts/search_oof_postprocess.py --config configs/canonical_baseline.yaml --folds 0,1,2,3
```

Training logs are written by the command runner you use. Model histories and validation exports are written under `outputs/` according to the active config:

```text
outputs/logs/
```

Before launching training, check the loaded Python environment:

```bash
python scripts/check_environment.py --strict
```

### Alvis

Alvis uses Slurm and software modules. The maintained Slurm templates call
`scripts/train_all_folds.sh`, which now defaults to the canonical baseline.
Find a CUDA PyTorch module on Alvis:

```bash
module spider PyTorch-bundle
```

Then load the selected module inside your Slurm job before running the Stage1, Stage2, and OOF commands.

### 5. Export validation results manually (optional)

```bash
python scripts/evaluate_val.py --config configs/canonical_baseline.yaml --fold 0
```

Use this when you want to rerun validation export separately. It writes:

- `val_metrics.json`
- `val_per_image.csv`
- `val_postprocess_search.csv`

To render false-positive, false-negative, and worst-defect overlays for a fold:

```bash
python scripts/visualize_error_analysis.py --config configs/canonical_baseline.yaml --fold 0
```

This writes `summary.csv`, per-image overlay panels, and contact sheets under:

```text
outputs/experiments/canonical_baseline/stage2/fold{fold}/error_analysis/
```

### 6. Run holdout inference

```bash
python scripts/infer_holdout_ensemble.py --config configs/canonical_baseline.yaml --folds 0,1,2,3
```

This step requires:

- `outputs/experiments/canonical_baseline/stage2/fold{fold}/best_stage2.pt` for all requested folds
- `outputs/experiments/canonical_baseline/stage2/oof_global_postprocess.json`

It writes holdout outputs under:

```text
outputs/experiments/canonical_baseline/stage2/holdout_ensemble/
|- prob_maps/
|- raw_binary_masks/
|- masks/
`- inference_summary.csv
```

## Run Other Folds

Replace `--fold 0` with the target fold:

```bash
python scripts/train_stage1.py --config configs/canonical_baseline.yaml --fold 1
python scripts/train_stage2.py --config configs/canonical_baseline.yaml --fold 1
python scripts/evaluate_val.py --config configs/canonical_baseline.yaml --fold 1
python scripts/infer_holdout_ensemble.py --config configs/canonical_baseline.yaml --folds 0,1,2,3
```

Available folds:

- `0`
- `1`
- `2`
- `3`

## Output Directory

Default output layout from the canonical config:

```text
outputs/experiments/canonical_baseline/
|- stage1/
|  `- fold{fold}/
|     |- best_stage1.pt
|     |- last_stage1.pt
|     |- history.csv
|     `- replay/
|- stage2/
|  `- fold{fold}/
|     |- best_stage2.pt
|     |- last_stage2.pt
|     |- history.csv
|     |- val_metrics.json
|     |- val_per_image.csv
|     |- val_postprocess_search.csv
|     `- hard_normal/
`- stage2/holdout_ensemble/
```
