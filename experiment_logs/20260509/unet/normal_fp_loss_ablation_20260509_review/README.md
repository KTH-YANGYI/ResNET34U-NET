# normal_fp_loss_weight ablation review

Created: 2026-05-09T19:04:18.452599

Purpose: compare Stage2 `normal_fp_loss_weight` values after selecting `pos_weight=12` as the current baseline. These runs test whether a small auxiliary penalty on normal false-positive pixels reduces normal FPR without losing too much defect segmentation quality.

## Variants

| variant | normal_fp_loss_weight | source experiment | config |
|---|---:|---|---|
| pw12_fp003 | 0.03 | `outputs/experiments/p0_a40_e50_bs48_pw12_fp003_20260509` | `configs/stage2_p0_a40_e50_bs48_pw12_fp003.yaml` |
| pw12_fp005 | 0.05 | `outputs/experiments/p0_a40_e50_bs48_pw12_fp005_20260509` | `configs/stage2_p0_a40_e50_bs48_pw12_fp005.yaml` |

## Final Pooled OOF Metrics

| run | defect_dice | defect_iou | defect_image_recall | normal_fpr | normal_fp_count | threshold/min_area |
|---|---:|---:|---:|---:|---:|---|
| baseline pw12 | 0.756876 | 0.633878 | 0.995261 | 0.000000 | 0 | 0.80/24 |
| pw12_fp003 | 0.741675 | 0.623785 | 0.966825 | 0.067797 | 4 | 0.82/8 |
| pw12_fp005 | 0.710942 | 0.589482 | 0.947867 | 0.000000 | 0 | 0.82/16 |

## Comparison

- `pw12_fp003`: Dice delta vs baseline = -0.015201; normal_fpr delta = +0.067797; selected threshold/min_area = 0.82/8.
- `pw12_fp005`: Dice delta vs baseline = -0.045933; normal_fpr delta = +0.000000; selected threshold/min_area = 0.82/16.
- In this run, neither normal_fp_loss_weight variant improves the current `pos_weight=12` baseline OOF Dice. `fp005` keeps normal_fpr at 0 but loses much more Dice/recall; `fp003` also loses Dice and allows 4 normal false positives.

## Hard-normal cap observations

- Fold logs show repeated `hard normal target was capped by hard_normal_max_repeats_per_epoch` messages. This is expected: the cap is active and prevents small hard pools from being replayed excessively.
- Some folds reached the full hard-normal target when the hard pool was large enough; small pools were capped to roughly two repeats per hard-normal sample.

## Included materials

- `logs/p0_a40_normal_fp_loss_ablation_20260509.log`: main batch log.
- `{variant}/config/`: Stage2 config used by each variant.
- `{variant}/EXPERIMENT.md`: per-experiment purpose note.
- `{variant}/logs/`: fold logs and OOF postprocess logs.
- `{variant}/stage2/`: pooled OOF JSON, threshold search CSV, per-image OOF CSV, and per-fold history/validation files copied when present.
- `baseline_pw12/oof_global_postprocess.json`: completed `pos_weight=12` baseline used for comparison.

Original experiment directories were preserved; files here are copied for review convenience.
