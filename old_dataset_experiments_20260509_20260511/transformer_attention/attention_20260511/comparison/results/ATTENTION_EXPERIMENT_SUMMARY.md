# Attention Experiment Summary

OOF results use the pooled validation threshold/min-area search. Holdout results, when available, must use the frozen OOF post-processing setting and must not tune on holdout.

## OOF Metrics

| experiment | defect_dice | defect_iou | defect_image_recall | normal_fp_count | normal_fpr | normal_fp_pixel_sum | normal_largest_fp_area_max | pixel_precision_defect_macro | pixel_recall_defect_macro | pixel_f1_defect_macro | pixel_auprc_all_labeled | component_recall_3px | component_precision_3px | component_f1_3px | boundary_f1_3px | threshold | min_area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline ResNet34-U-Net pw12 | 0.753116 | 0.629560 | 0.990521 | 1 | 0.016949 | 28.000000 | 28.000000 | 0.746750 | 0.825663 | 0.753116 | 0.872378 | 0.970774 | 0.849831 | 0.879124 | 0.751804 | 0.780000 | 16 |
| Transformer bottleneck d1 | 0.756183 | 0.634339 | 0.990521 | 1 | 0.016949 | 54.000000 | 54.000000 | 0.761185 | 0.815341 | 0.756183 | 0.870836 | 0.970774 | 0.854604 | 0.881648 | 0.761547 | 0.800000 | 16 |
| TBN d1 + hard-negative prototype attention | 0.757016 | 0.636425 | 0.981043 | 1 | 0.016949 | 104.000000 | 104.000000 | 0.759936 | 0.824406 | 0.757016 | 0.874674 | 0.960900 | 0.882622 | 0.890890 | 0.756515 | 0.800000 | 32 |
| Decoder skip attention gate d4+d3 | 0.742251 | 0.624099 | 0.971564 | 2 | 0.033898 | 234.000000 | 202.000000 | 0.767554 | 0.794401 | 0.742251 | 0.870284 | 0.951027 | 0.847292 | 0.857330 | 0.730087 | 0.860000 | 8 |

## OOF Deltas

| experiment | comparison | delta_defect_dice | delta_defect_iou | delta_defect_image_recall | delta_normal_fp_count | delta_normal_fpr | delta_pixel_f1_defect_macro | delta_pixel_auprc_all_labeled | delta_component_recall_3px | delta_boundary_f1_3px |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01_tbn_d1 | vs_baseline | 0.003067 | 0.004779 | 0.000000 | 0.000000 | 0.000000 | 0.003067 | -0.001542 | 0.000000 | 0.009743 |
| 02_tbn_d1_hnproto | vs_baseline | 0.003901 | 0.006865 | -0.009479 | 0.000000 | 0.000000 | 0.003901 | 0.002296 | -0.009874 | 0.004710 |
| 02_tbn_d1_hnproto | vs_01_tbn_d1 | 0.000833 | 0.002086 | -0.009479 | 0.000000 | 0.000000 | 0.000833 | 0.003838 | -0.009874 | -0.005032 |
| 03_skipgate_d4d3 | vs_baseline | -0.010864 | -0.005461 | -0.018957 | 1.000000 | 0.016949 | -0.010864 | -0.002094 | -0.019747 | -0.021717 |

## Frozen Holdout Metrics

| experiment | defect_dice | defect_iou | defect_image_recall | normal_fp_count | normal_fpr | normal_fp_pixel_sum | normal_largest_fp_area_max | pixel_precision_defect_macro | pixel_recall_defect_macro | pixel_f1_defect_macro | pixel_auprc_all_labeled | component_recall_3px | component_precision_3px | component_f1_3px | boundary_f1_3px | threshold | min_area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline ResNet34-U-Net pw12 | 0.750089 | 0.626415 | 0.961538 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.799652 | 0.776157 | 0.750089 | 0.923386 | 0.945513 | 0.935897 | 0.911264 | 0.759654 | 0.780000 | 16 |
| Transformer bottleneck d1 | 0.752313 | 0.630033 | 0.961538 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.818820 | 0.760920 | 0.752313 | 0.923817 | 0.945513 | 0.939103 | 0.913095 | 0.764477 | 0.800000 | 16 |
| TBN d1 + hard-negative prototype attention | 0.751929 | 0.629719 | 0.961538 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.806672 | 0.771002 | 0.751929 | 0.921651 | 0.935897 | 0.953526 | 0.915842 | 0.760800 | 0.800000 | 32 |
| Decoder skip attention gate d4+d3 | 0.722522 | 0.604266 | 0.942308 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.835726 | 0.723090 | 0.722522 | 0.924318 | 0.907051 | 0.910256 | 0.862546 | 0.736430 | 0.860000 | 8 |

## Notes

- `stage2_score` is intentionally excluded because it is a checkpoint/post-processing selection score, not a thesis endpoint.
- The strictest claim should require improved or non-degraded Dice/IoU while keeping normal false positives at baseline level.
- For `02_tbn_d1_hnproto`, the `vs_01_tbn_d1` row isolates the added value of hard-normal prototype attention beyond the transformer bottleneck.
