# UNET Two-Stage

Two-stage defect segmentation pipeline:

1. Stage1 trains a patch-level model on cropped positive and negative patches.
2. Stage2 loads the best Stage1 checkpoint and trains a full-image segmentation model.
3. Stage2 selects its best checkpoint with the same validation post-processing search used for final reporting.
4. Validation export and holdout inference/evaluation reuse the post-processing thresholds selected on the validation set.

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

The planned decoder self-attention ablation is kept separate at
`configs/experiments/self_attention_d4d3.yaml` and summarized by
`experiment_sets/self_attention_20260515.yaml`.

The prototype cross-attention ablation is kept at
`configs/experiments/prototype_cross_attention.yaml`. It requires a prototype
bank built from the canonical Stage1 checkpoint before Stage2 training.

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
- `UNET_two_stage/` and `dataset_crack_normal_unet_811/` are sibling directories under the same parent directory.
- Generated paths under `manifests/` and `outputs/` are resolved relative to the project root.
- The active dataset uses an explicit `train`/`val`/`test` split. There is no cross-validation.

## Dataset Layout

`scripts/prepare_samples.py` expects the split dataset under
`../dataset_crack_normal_unet_811/` by default:

```text
parent_dir/
|- UNET_two_stage/
`- dataset_crack_normal_unet_811/

dataset_crack_normal_unet_811/
|- train/
|  |- images/crack/
|  |- images/normal/
|  |- masks/crack/
|  `- masks/normal/
|- val/
|  |- images/crack/
|  |- images/normal/
|  |- masks/crack/
|  `- masks/normal/
|- test/
|  |- images/crack/
|  |- images/normal/
|  |- masks/crack/
|  `- masks/normal/
|- manifest.csv
`- summary.json
```

Notes:

- `crack` images use the provided binary masks.
- `normal` images use the provided all-black masks.
- `test` is held out from Stage1/Stage2 training and validation.
- Legacy unsplit ROI datasets live only in the frozen archive.

### Dataset Roles And Counts

For the current `dataset_crack_normal_unet_811` snapshot:

| split | crack | normal | total | role |
| --- | ---: | ---: | ---: | --- |
| `train` | 371 | 274 | 645 | Stage1/Stage2 training |
| `val` | 47 | 34 | 81 | validation and threshold selection |
| `test` | 45 | 35 | 80 | final holdout inference/evaluation |
| total | 463 | 343 | 806 | |

The classes are used as follows:

- `crack`: labeled defect images with binary masks.
- `normal`: negative images used for training and validation.
- `broken`: excluded from the active dataset.

Stage1 expands the `train` and `val` images into patch indexes; the Stage1
`batch_size` is a patch batch size, while the Stage2 `batch_size` is a
full-image batch size. Stage2 uses native image sizes by default; mixed-size
images are batched by original tensor `HxW` with `stage2.batch_size_by_image_size`.

## Pipeline

### 1. Generate manifests

```bash
python scripts/prepare_samples.py
```

If the dataset is somewhere else, pass it explicitly:

```bash
python scripts/prepare_samples.py --dataset-root /path/to/dataset_crack_normal_unet_811
```

This step scans the dataset and writes:

- `manifests/samples.csv`
- `manifests/samples_summary.json`

Rows with `split=train` are used for training, rows with `split=val` are used
for validation, and rows with `split=test` are consumed by holdout inference.

Run this step again on the training server. `manifests/samples.csv` stores absolute image and mask paths for the current machine, so a local copy should not be reused after moving to a server.

### 2. Build Stage1 patch index

```bash
python scripts/build_patch_index.py --config configs/canonical_baseline.yaml
```

Run this after `prepare_samples.py` on the server so patch indexes also point to the server paths.

This step writes:

- `manifests/stage1_train_index.csv`
- `manifests/stage1_val_index.csv`
- `manifests/stage1_patch_summary.json`

### 3. Train Stage1

```bash
python scripts/train_stage1.py --config configs/canonical_baseline.yaml
```

Default Stage1 save directory from `configs/canonical_baseline.yaml`:

```text
outputs/experiments/canonical_baseline/stage1/
```

### 4. Train Stage2

```bash
python scripts/train_stage2.py --config configs/canonical_baseline.yaml
```

Stage2 uses:

- `manifests/samples.csv`
- `outputs/experiments/canonical_baseline/stage1/best_stage1.pt`

Default Stage2 save directory from `configs/canonical_baseline.yaml`:

```text
outputs/experiments/canonical_baseline/stage2/
```

By default, `train_stage2.py` also runs validation export automatically after training because `auto_evaluate_after_train: true` in `configs/canonical_baseline.yaml`.

Stage2 hard normal replay keeps the normal-image budget fixed. With `stage2_hard_normal_ratio: 0.40`, about 40% of the normal samples in an epoch come from the mined hard-normal pool once that pool exists, and the rest stay randomly sampled normals.

### Run The Canonical Pipeline

To reproduce the single canonical run, either run the commands above explicitly
or use the wrapper:

```bash
bash scripts/train_pipeline.sh --gpu 0
```

To use a 4-GPU A100 allocation on one node:

```bash
bash scripts/train_pipeline.sh --gpus 0,1,2,3
```

The canonical config enables `torch.nn.DataParallel`; if only one GPU is
visible, the scripts automatically fall back to single-GPU execution.

The explicit commands are:

```bash
python scripts/prepare_samples.py
python scripts/build_patch_index.py --config configs/canonical_baseline.yaml
python scripts/train_stage1.py --config configs/canonical_baseline.yaml
python scripts/train_stage2.py --config configs/canonical_baseline.yaml
```

Training logs are written by the command runner you use. Model histories and validation exports are written under `outputs/` according to the active config:

```text
outputs/logs/
```

Before launching training, check the loaded Python environment:

```bash
python scripts/check_environment.py --strict
```

### Running Stage2 Experiment Variants

Experiment configs under `configs/experiments/` inherit the canonical baseline.
They are intended to reuse the canonical Stage1 checkpoint and run Stage2 only:

```bash
bash scripts/train_pipeline.sh \
  --gpus 0,1,2,3 \
  --config configs/experiments/self_attention_d4d3.yaml \
  --stage2-only
```

The wrapper refuses to run Stage1 with a non-canonical experiment config unless
you pass `--allow-experiment-stage1`, which avoids accidentally overwriting or
mixing the canonical baseline Stage1 outputs.

For prototype cross-attention, first build the prototype bank:

```bash
python scripts/build_stage1_prototype_bank.py \
  --config configs/experiments/prototype_cross_attention.yaml
```

Then run Stage2:

```bash
bash scripts/train_pipeline.sh \
  --gpus 0,1,2,3 \
  --config configs/experiments/prototype_cross_attention.yaml \
  --stage2-only
```

### Alvis

Alvis uses Slurm and software modules. The maintained Slurm templates call
`scripts/train_pipeline.sh`, which defaults to the canonical baseline.
`scripts/alvis_train_a100.sbatch` requests 4 A100 GPUs and exposes them as
`CUDA_VISIBLE_DEVICES=0,1,2,3`.
Find a CUDA PyTorch module on Alvis:

```bash
module spider PyTorch-bundle
```

Then load the selected module inside your Slurm job before running the Stage1, Stage2, and holdout commands.

### 5. Export validation results manually (optional)

```bash
python scripts/evaluate_val.py --config configs/canonical_baseline.yaml
```

Use this when you want to rerun validation export separately. It writes:

- `val_metrics.json`
- `val_per_image.csv`
- `val_postprocess_search.csv`

To render false-positive, false-negative, and worst-defect overlays:

```bash
python scripts/visualize_error_analysis.py --config configs/canonical_baseline.yaml
```

This writes `summary.csv`, per-image overlay panels, and contact sheets under:

```text
outputs/experiments/canonical_baseline/stage2/error_analysis/
```

### 6. Run holdout inference/evaluation

```bash
python scripts/infer_holdout.py --config configs/canonical_baseline.yaml
```

This step requires:

- `outputs/experiments/canonical_baseline/stage2/best_stage2.pt`
- `outputs/experiments/canonical_baseline/stage2/val_metrics.json`

Because the current test split has masks, this step writes both predictions and
quantitative test metrics under:

```text
outputs/experiments/canonical_baseline/stage2/holdout/
|- prob_maps/
|- raw_binary_masks/
|- masks/
|- inference_summary.csv
|- holdout_metrics.json
|- holdout_per_image.csv
`- holdout_group_metrics.csv
```

## Output Directory

Default output layout from the canonical config:

```text
outputs/experiments/canonical_baseline/
|- stage1/
|  |- best_stage1.pt
|  |- last_stage1.pt
|  |- history.csv
|  `- replay/
`- stage2/
   |- best_stage2.pt
   |- last_stage2.pt
   |- history.csv
   |- val_metrics.json
   |- val_per_image.csv
   |- val_postprocess_search.csv
   |- hard_normal/
   `- holdout/
      |- holdout_metrics.json
      |- holdout_per_image.csv
      `- holdout_group_metrics.csv
```
