# Experiment Summary - 2026-05-09

> Archive note: this file summarizes historical old-dataset runs. Current active
> training uses `configs/canonical_baseline.yaml`; the old configs/scripts
> referenced below are frozen under `old_dataset_experiments_20260509_20260511/`.

This document summarizes the U-Net and YOLO experiments run on Alvis on 2026-05-09, where the logs were kept, what each experiment was intended to test, the observed results, and which parts are suitable for the thesis.

The lightweight logs and review artifacts copied into this repository are under:

- `experiment_logs/20260509/unet/`
- `experiment_logs/20260509/yolo/`

The original server workspaces are:

- U-Net: `/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/UNET_two_stage`
- YOLO: `/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/yolo`

Only text artifacts were copied to GitHub: README files, configs, scripts, logs, CSV/JSON summaries, OOF metrics, and small analysis tables. Checkpoints, prediction images, bbox visualization images, and other large generated artifacts were intentionally not copied into the repository.

## U-Net Experiments

### Code and training changes

The U-Net work focused on making Stage2 training more stable and then testing targeted ablations.

Implemented changes:

- Stage2 checkpoint selection now uses fixed train-time `threshold/min_area` instead of searching threshold and min_area every epoch.
- Hard normal replay now has `hard_normal_max_repeats_per_epoch`, preventing a tiny hard-normal pool from being replayed dozens of times.
- Stage1 train loader uses a different seed each epoch, reducing repeated augmentation patterns.
- Frozen encoder mode also freezes encoder BatchNorm running statistics.
- Error analysis visualization script was added: `scripts/visualize_error_analysis.py`.
- PatchDataset worker-local cache was added as a data-loading optimization.
- Deep supervision support was added.
- Boundary auxiliary loss support was added.

These code changes were pushed earlier in commits:

- `956dda9 Stabilize stage2 training selection`
- `015d3d8 Add second-stage ablation configs`
- `cefa220 Add normal FP loss ablation runner`
- `a2f18c6 Add third-batch segmentation experiments`

### U-Net log locations

Primary copied review folders:

- `experiment_logs/20260509/unet/pos_weight_ablation_20260509_review/`
- `experiment_logs/20260509/unet/normal_fp_loss_ablation_20260509_review/`
- `experiment_logs/20260509/unet/third_batch_segmentation_20260509_review/`

Important files:

- `experiment_logs/20260509/unet/pos_weight_ablation_20260509_review/README.md`
- `experiment_logs/20260509/unet/normal_fp_loss_ablation_20260509_review/README.md`
- `experiment_logs/20260509/unet/third_batch_segmentation_20260509_review/README.md`
- `experiment_logs/20260509/unet/third_batch_segmentation_20260509_review/unet_oof_comparison.csv`

Original server logs remain under:

- `outputs/experiments/p0_a40_second_batch_20260509.log`
- `outputs/experiments/p0_a40_normal_fp_loss_ablation_20260509.log`
- `outputs/experiments/p0_a40_third_batch_20260509.log`

Original server experiment directories remain preserved under `outputs/experiments/`.

### Pos weight ablation

Purpose: test Stage2 positive class weighting under the stabilized Stage2 checkpoint/post-processing flow.

Copied logs:

- `experiment_logs/20260509/unet/pos_weight_ablation_20260509_review/`

Results:

| variant | pos_weight | Dice | IoU | defect recall | normal FPR | threshold | min_area | normal FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pw6 | 6 | 0.755910 | 0.635336 | 0.990521 | 0.016949 | 0.74 | 8 | 1 |
| pw8 | 8 | 0.751339 | 0.631859 | 0.981043 | 0.016949 | 0.76 | 24 | 1 |
| pw12 | 12 | 0.756876 | 0.633878 | 0.995261 | 0.000000 | 0.80 | 24 | 0 |

Conclusion:

- `pos_weight=12` is the best balanced U-Net setting.
- It gives the best Dice, very high defect recall, and zero normal false-positive rate in pooled OOF evaluation.
- This should be included in the thesis ablation table.

### normal_fp_loss_weight ablation

Purpose: after selecting `pos_weight=12`, test whether an additional small loss term on normal false-positive pixels reduces normal false positives without hurting segmentation quality.

Copied logs:

- `experiment_logs/20260509/unet/normal_fp_loss_ablation_20260509_review/`

Results:

| variant | normal_fp_loss_weight | Dice | IoU | defect recall | normal FPR | normal FP | threshold/min_area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline pw12 | 0 | 0.756876 | 0.633878 | 0.995261 | 0.000000 | 0 | 0.80/24 |
| pw12_fp003 | 0.03 | 0.741675 | 0.623785 | 0.966825 | 0.067797 | 4 | 0.82/8 |
| pw12_fp005 | 0.05 | 0.710942 | 0.589482 | 0.947867 | 0.000000 | 0 | 0.82/16 |

Conclusion:

- Neither value improved the baseline.
- `0.03` reduced segmentation quality and introduced normal false positives.
- `0.05` kept normal FPR at zero but sacrificed too much Dice and recall.
- This is a useful negative ablation for the thesis.

### Third-batch segmentation experiments

Purpose: test whether architectural/training extensions can improve the selected `pos_weight=12` baseline.

Copied logs:

- `experiment_logs/20260509/unet/third_batch_segmentation_20260509_review/`
- Main comparison table: `experiment_logs/20260509/unet/third_batch_segmentation_20260509_review/unet_oof_comparison.csv`

Results:

| variant | Dice | IoU | defect recall | normal FPR | normal FP | threshold | min_area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pw12 baseline | 0.756876 | 0.633878 | 0.995261 | 0.000000 | 0 | 0.80 | 24 |
| deep supervision | 0.752240 | 0.637742 | 0.957346 | 0.016949 | 1 | 0.86 | 48 |
| boundary auxiliary | 0.755884 | 0.633033 | 0.995261 | 0.033898 | 2 | 0.82 | 0 |

Conclusion:

- Deep supervision slightly improved IoU, but Dice and defect recall dropped and one normal false positive appeared.
- Boundary auxiliary loss did not improve Dice/IoU and increased normal false positives.
- PatchDataset cache is a runtime/data-loading improvement, not a metric-changing model result.
- The `pos_weight=12` baseline remains the best U-Net model.

### U-Net thesis relevance

Recommended thesis content:

- Main method: two-stage U-Net training with hard normal replay and global post-processing.
- Main result: `pos_weight=12` Stage2 baseline.
- Ablation table: `pos_weight`, `normal_fp_loss_weight`, deep supervision, boundary auxiliary loss.
- Implementation details: fixed Stage2 checkpoint threshold, hard-normal replay cap, per-epoch loader seed, frozen BatchNorm stats.
- Qualitative analysis: error analysis visualizations can be used for examples of TP/FP/FN behavior.

Current U-Net bottleneck:

- The strongest U-Net setting is already very recall-heavy and has zero normal FPR in OOF.
- Extra losses and auxiliary heads did not clearly improve the balanced objective.
- The remaining improvement space is likely not from broad architecture/loss additions, but from better data labeling, more data, or more targeted error-mode analysis.

## YOLO Experiments

YOLO was explored as a detector-style baseline after the U-Net experiments.

Important caveat:

- The YOLO dataset used here was converted from the U-Net manifest and masks.
- Source images are symlinks into `dataset0505_crop640_roi`.
- This is a crack/normal detection task, not a true crack/broken two-class training run.
- The `broken` images are in `predict_broken` only and were not used as labeled training examples.

YOLO copied logs:

- `experiment_logs/20260509/yolo/logs/`
- `experiment_logs/20260509/yolo/runs/yolo_strategy_review_20260509/YOLO_EXPERIMENTS_EXPLAINED.md`
- `experiment_logs/20260509/yolo/runs/yolo_strategy_review_20260509/yolo_experiment_explanation_metrics.csv`
- `experiment_logs/20260509/yolo/runs/yolo_strategy_review_20260509/yolo_experiment_explanation_metrics.json`
- `experiment_logs/20260509/yolo/runs/yolo11n_crack_broken_e80_2gpu_20260509/conf_sweep_20260509/`
- `experiment_logs/20260509/yolo/runs/yolo11n_crack_broken_e140_2gpu_baseline_20260509/error_inspection_20260509/`
- `experiment_logs/20260509/yolo/runs/yolo11n_crack_broken_e140_2gpu_baseline_20260509/nms_sweep_20260509/`

### YOLO dataset

Actual YOLO dataset directories:

- Baseline: `/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/yolo/yolo_data_set`
- Tight bbox variant: `/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/yolo/yolo_data_set_pad0_min8_20260509`

Baseline split:

| split | images | crack boxes | normal empty labels |
| --- | ---: | ---: | ---: |
| train | 200 | 157 | 43 |
| val | 70 | 54 | 16 |
| test | 67 | 52 | 15 |

The image files in `images/` are symlinks, so they may appear as around `108B` in file listings. The actual source PNG files are normal 640x640 RGB images under `dataset0505_crop640_roi`.

### YOLO runs

Metrics below are recalculated from each `results.csv` using true best mAP50-95 unless noted otherwise.

| run | purpose | best mAP50-95 | recall at best mAP50-95 | best mAP50 | status |
| --- | --- | ---: | ---: | ---: | --- |
| yolo11n e80 4GPU | test 4GPU DDP | - | - | - | DDP stuck, stopped |
| yolo11n e80 2GPU | baseline | 0.59920 | 0.85185 | 0.92936 | completed |
| yolo11n e80 pad0/min8 | tight bbox test | 0.56471 | 0.85185 | 0.92418 | completed, worse |
| yolo26n e80 | larger model test | 0.55857 | 0.75717 | 0.89434 | completed, worse |
| yolo11n e140 | longer training | 0.60735 | 0.84212 | 0.93251 | completed, best mAP50-95 |
| yolo11n e200 | longer training | 0.60550 | 0.87037 | 0.93687 | completed, no material gain |
| yolo11n e140 img832 | higher resolution | 0.60104 | 0.79630 | 0.94454 | completed, best mAP50 only |

Conclusion:

- The best detector by localization-quality metric `mAP50-95` is `yolo11n_crack_broken_e140_2gpu_baseline_20260509`.
- `img832` gives the highest mAP50, but not the best mAP50-95; it is better at coarse hits, not precise localization.
- `yolo26n` did not help, so the bottleneck is not simply model capacity.
- Normal-image false positives were not the main YOLO issue. Label checks found no invalid/tiny/duplicate labels. Remaining YOLO errors are mostly duplicate predictions and localization instability.

### YOLO thesis relevance

YOLO can be used as a supplementary detector baseline, but it should be described carefully:

- It is a mask-to-bbox crack detector baseline on crop ROI data.
- It is not a full broken/crack detector because broken examples were not labeled for training.
- It is less central than the U-Net result because the main task is pixel-level defect segmentation.

Recommended thesis usage:

- Include YOLO as an auxiliary baseline or appendix experiment.
- Use it to justify why pixel-level U-Net remains the main model.
- If used in the main text, clearly state the data conversion and class limitations.

## Overall Recommendation

For the thesis, the strongest main story is:

1. Two-stage U-Net with hard normal replay.
2. Stabilized Stage2 training and global post-processing.
3. `pos_weight=12` as the best U-Net configuration.
4. Negative ablations showing that normal FP loss, deep supervision, and boundary auxiliary loss did not improve the balanced result.
5. YOLO as a detector-style comparison, with clear caveats about dataset construction and task mismatch.

The best current U-Net model remains:

- `configs/stage2_p0_a40_e50_bs48_pw12.yaml`
- Original server output: `outputs/experiments/p0_a40_e50_bs48_pw12_20260509`
- OOF result: Dice `0.756876`, IoU `0.633878`, defect recall `0.995261`, normal FPR `0.000000`

The best current YOLO detector baseline is:

- `yolo11n_crack_broken_e140_2gpu_baseline_20260509`
- True best mAP50-95 `0.60735`, recall at that epoch `0.84212`
