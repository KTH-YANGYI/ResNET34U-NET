# Third-batch segmentation review

Purpose: collect the third-batch U-Net segmentation experiments and compare them against the completed pos_weight and normal_fp_loss_weight ablations.

## Included materials

- deep_supervision/: config, experiment note, fold logs, OOF log/result files, and per-fold history/validation exports.
- boundary_aux/: config, experiment note, fold logs, OOF log/result files, and per-fold history/validation exports.
- shared/: third-batch runner, stage2 runner, PatchDataset cache note/config, and main batch log.
- unet_oof_comparison.csv: compact metric table across baseline, ablations, and third-batch runs.

## OOF metrics

| variant | Dice | IoU | recall | normal_fpr | normal_fp | threshold | min_area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pw6 | 0.755910 | 0.635336 | 0.990521 | 0.016949 | 1 | 0.740000 | 8 |
| pw8 | 0.751339 | 0.631859 | 0.981043 | 0.016949 | 1 | 0.760000 | 24 |
| pw12_baseline | 0.756876 | 0.633878 | 0.995261 | 0.000000 | 0 | 0.800000 | 24 |
| pw12_fp003 | 0.741675 | 0.623785 | 0.966825 | 0.067797 | 4 | 0.820000 | 8 |
| pw12_fp005 | 0.710942 | 0.589482 | 0.947867 | 0.000000 | 0 | 0.820000 | 16 |
| deep_supervision | 0.752240 | 0.637742 | 0.957346 | 0.016949 | 1 | 0.860000 | 48 |
| boundary_aux | 0.755884 | 0.633033 | 0.995261 | 0.033898 | 2 | 0.820000 | 0 |

## Notes

- PatchDataset worker-local cache is a runtime/data-loading optimization only; it does not change Stage2 predictions unless Stage1 is rerun with configs/stage1_p0_a40_cache.yaml.
- hard_normal_max_repeats_per_epoch capped replay throughout Stage2. It prevented small hard pools from being repeated dozens of times; cap warnings are expected and indicate the guard is active.
- pos_weight=12 baseline remains the strongest balanced U-Net point: Dice 0.756876, recall 0.995261, normal_fpr 0.000000, threshold/min_area 0.800000/24.
- deep_supervision improved IoU slightly to 0.637742, but lowered Dice/recall and introduced one normal false positive at threshold/min_area 0.860000/48.
- boundary_aux stayed near baseline recall but has lower Dice than baseline and two normal false positives at threshold/min_area 0.820000/0.

## U-Net optimization-space review

The most reliable U-Net configuration is still the pos_weight=12 baseline. normal_fp_loss_weight worsened Dice/recall, deep supervision traded recall for a small IoU gain, and boundary auxiliary loss did not reduce false positives enough to justify the extra head/loss. With the current OOF evidence, the next high-value direction is not another broad U-Net ablation; use YOLO as the next detector-style comparison and revisit U-Net only if YOLO exposes a clear failure mode such as localization sensitivity or systematic background false positives.
