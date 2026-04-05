# UNET Two-Stage

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

## Dataset Layout

The dataset is expected under:

```text
dataset_new/
├─ train/
│  ├─ images/
│  └─ masks/
├─ val/
├─ normal_crops_selected/
└─ 图片来源视频映射/
```

## Run Order

### 1. Generate manifests

```bash
python scripts/prepare_dataset.py
```

### 2. Build Stage1 patch index

```bash
python scripts/build_patch_index.py --config configs/stage1.yaml
```

### 3. Train Stage1

```bash
python scripts/train_stage1.py --config configs/stage1.yaml --fold 0
```

### 4. Train Stage2

```bash
python scripts/train_stage2.py --config configs/stage2.yaml --fold 0
```

### 5. Export validation results

```bash
python scripts/evaluate_val.py --config configs/stage2.yaml --fold 0
```

### 6. Run holdout inference

```bash
python scripts/infer_holdout.py --config configs/stage2.yaml --fold 0
```

### 7. Generate training plots

```bash
python scripts/plot_training_reports.py --root outputs
```

## Run Other Folds

Replace `--fold 0` with the target fold:

```bash
python scripts/train_stage1.py --config configs/stage1.yaml --fold 1
python scripts/train_stage2.py --config configs/stage2.yaml --fold 1
```

Available folds:

- `0`
- `1`
- `2`
- `3`

## Output Directory

Default outputs:

```text
outputs/
├─ stage1/fold{fold}/
├─ stage2/fold{fold}/
└─ reports/
```
