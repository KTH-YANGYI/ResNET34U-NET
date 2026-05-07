# UNET Two-Stage

Two-stage defect segmentation pipeline:

1. Stage1 trains a patch-level model on cropped positive and negative patches.
2. Stage2 loads the best Stage1 checkpoint and trains a full-image segmentation model.
3. Validation export and holdout inference reuse the post-processing thresholds selected on the validation set.

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
- Cross-validation is image-level random splitting; video mapping files are not used.

## Pipeline

### 1. Generate manifests

```bash
python scripts/prepare_samples.py
```

If the dataset is somewhere else, pass it explicitly:

```bash
python scripts/prepare_samples.py --dataset-root /path/to/dataset0505_crop640_roi
```

This step scans the dataset and writes:

- `manifests/samples.csv`
- `manifests/samples_summary.json`
- `generated_masks/{device}/crack/*.png`

Run this step again on the training server. `manifests/samples.csv` stores absolute image and mask paths for the current machine, so a local copy should not be reused after moving to a server.

### 2. Build Stage1 patch index

```bash
python scripts/build_patch_index.py --config configs/stage1.yaml
```

Run this after `prepare_samples.py` on the server so patch indexes also point to the server paths.

This step writes:

- `manifests/stage1_fold{fold}_train_index.csv`
- `manifests/stage1_fold{fold}_val_index.csv`
- `manifests/stage1_patch_summary.json`

### 3. Train Stage1

```bash
python scripts/train_stage1.py --config configs/stage1.yaml --fold 0
```

Default Stage1 save directory from `configs/stage1.yaml`:

```text
outputs/stage1/fold{fold}/
```

### 4. Train Stage2

```bash
python scripts/train_stage2.py --config configs/stage2.yaml --fold 0
```

Stage2 uses:

- `manifests/samples.csv`
- `outputs/stage1/fold{fold}/best_stage1.pt`

Default Stage2 save directory from `configs/stage2.yaml`:

```text
outputs/stage2/fold{fold}/
```

By default, `train_stage2.py` also runs validation export automatically after training because `auto_evaluate_after_train: true` in `configs/stage2.yaml`.

### Run All Folds

On a Linux training server, this runs preparation, Stage1, and Stage2 for folds `0,1,2,3`:

```bash
bash scripts/train_all_folds.sh --gpus 0
```

Use multiple GPUs to run folds in parallel:

```bash
bash scripts/train_all_folds.sh --gpus 0,1,2,3
```

Useful variants:

```bash
bash scripts/train_all_folds.sh --gpus 0,1 --skip-prepare
bash scripts/train_all_folds.sh --gpus 0 --stage1-only
bash scripts/train_all_folds.sh --gpus 0 --stage2-only --skip-prepare
bash scripts/train_all_folds.sh --gpus 0,1,2,3 --with-holdout
```

Logs are written to:

```text
outputs/logs/
```

Before launching training, check the loaded Python environment:

```bash
python scripts/check_environment.py --strict
```

### Alvis

Alvis uses Slurm and software modules. Do not install Python system-wide. Find a CUDA PyTorch module on Alvis:

```bash
module spider PyTorch-bundle
```

Edit `scripts/alvis_train.sbatch` and replace:

```text
#SBATCH -A CHANGE_ME
```

with your NAISS/C3SE project account. Then submit a job:

```bash
sbatch scripts/alvis_train.sbatch
```

If the default module name is not available, pass the exact module version shown by `module spider`:

```bash
sbatch --export=ALL,PYTORCH_MODULE=PyTorch-bundle/CHANGE_ME scripts/alvis_train.sbatch
```

If optional imports are missing, training can still run. For example, `PyYAML` and `tqdm` are optional because the project has a built-in config parser and a no-progress fallback. You can still load extra modules if you want them:

```bash
module spider PyYAML
EXTRA_MODULES="PyYAML/CHANGE_ME" sbatch --export=ALL scripts/alvis_train.sbatch
```

The included `scripts/alvis_train.sbatch` requests four A40 GPUs by default:

```text
#SBATCH --gpus-per-node=A40:4
```

That means the default submission runs four folds in parallel:

```bash
sbatch scripts/alvis_train.sbatch
```

To request a different GPU type or count, edit the `--gpus-per-node` line. Examples:

```text
#SBATCH --gpus-per-node=T4:1
#SBATCH --gpus-per-node=T4:4
#SBATCH --gpus-per-node=A40:4
#SBATCH --gpus-per-node=A100:4
```

If you change the number of GPUs, also set `CUDA_FOLD_GPUS` accordingly. For example:

```bash
CUDA_FOLD_GPUS=0,1,2,3 sbatch --export=ALL scripts/alvis_train.sbatch
```

Check job state and logs:

```bash
squeue -u "$USER"
tail -f slurm-unet_two_stage_JOBID.out
tail -f outputs/logs/stage1_fold0.log
```

### 5. Export validation results manually (optional)

```bash
python scripts/evaluate_val.py --config configs/stage2.yaml --fold 0
```

Use this when you want to rerun validation export separately. It writes:

- `val_metrics.json`
- `val_per_image.csv`
- `val_postprocess_search.csv`

### 6. Run holdout inference

```bash
python scripts/infer_holdout.py --config configs/stage2.yaml --fold 0
```

This step requires:

- `outputs/stage2/fold{fold}/best_stage2.pt`
- `outputs/stage2/fold{fold}/val_metrics.json`

It writes holdout outputs under:

```text
outputs/stage2/fold{fold}/holdout/
|- prob_maps/
|- raw_binary_masks/
|- masks/
`- inference_summary.csv
```

## Run Other Folds

Replace `--fold 0` with the target fold:

```bash
python scripts/train_stage1.py --config configs/stage1.yaml --fold 1
python scripts/train_stage2.py --config configs/stage2.yaml --fold 1
python scripts/evaluate_val.py --config configs/stage2.yaml --fold 1
python scripts/infer_holdout.py --config configs/stage2.yaml --fold 1
```

Available folds:

- `0`
- `1`
- `2`
- `3`

## Output Directory

Default output layout from the config files:

```text
outputs/
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
|     |- hard_normal/
|     `- holdout/
```
