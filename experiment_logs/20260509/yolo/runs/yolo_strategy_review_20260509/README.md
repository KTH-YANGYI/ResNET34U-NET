# YOLO strategy review 2026-05-09

## Completed runs

- yolo11n_e80_baseline: best mAP50=0.92936, best mAP50-95=0.57283, best_epoch=68 (yolo11n_crack_broken_e80_2gpu_20260509)
- yolo11n_pad0_min8: best mAP50=0.92418, best mAP50-95=0.56471, best_epoch=73 (yolo11n_crack_broken_e80_2gpu_pad0_min8_20260509)
- yolo26n_e80_baseline: best mAP50=0.89434, best mAP50-95=0.55857, best_epoch=61 (yolo26n_crack_broken_e80_2gpu_baseline_20260509)
- yolo11n_e140_baseline: best mAP50=0.93251, best mAP50-95=0.60022, best_epoch=135 (yolo11n_crack_broken_e140_2gpu_baseline_20260509)
- yolo11n_e200_baseline: best mAP50=0.93687, best mAP50-95=0.55164, best_epoch=137 (yolo11n_crack_broken_e200_2gpu_baseline_20260509)
- yolo11n_e140_img832: best mAP50=0.94454, best mAP50-95=0.58694, best_epoch=117 (yolo11n_crack_broken_e140_img832_2gpu_20260509)

## Decision

- Preferred YOLO checkpoint direction: `yolo11n_crack_broken_e140_2gpu_baseline_20260509`.
- It improved mAP50-95 over e80 and kept a better balance than yolo26n, e200, pad0/min8, and img832.
- img832 improved best mAP50 but reduced mAP50-95 and recall balance, so it is not preferred.
- Label inspection found no invalid/tiny/duplicate labels. Normal-image false positives were not the main issue; remaining errors are mostly duplicate predictions/localization.
- NMS sweep on e140 suggests NMS IoU around 0.55 is the best validation operating point.

## Useful analysis outputs

- `runs/yolo11n_crack_broken_e80_2gpu_20260509/conf_sweep_20260509`
- `runs/yolo11n_crack_broken_e140_2gpu_baseline_20260509/error_inspection_20260509`
- `runs/yolo11n_crack_broken_e140_2gpu_baseline_20260509/nms_sweep_20260509`
