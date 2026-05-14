# Pos Weight Ablation Review

Source workspace: /mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/UNET_two_stage
Batch log: logs/p0_a40_second_batch_20260509.log

## OOF Summary

| variant | pos_weight | defect_dice | defect_iou | defect_image_recall | normal_fpr | threshold | min_area | normal_fp_count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pw6 | 6 | 0.755910 | 0.635336 | 0.990521 | 0.016949 | 0.74 | 8 | 1 |
| pw8 | 8 | 0.751339 | 0.631859 | 0.981043 | 0.016949 | 0.76 | 24 | 1 |
| pw12 | 12 | 0.756876 | 0.633878 | 0.995261 | 0.000000 | 0.80 | 24 | 0 |

Best OOF defect_dice: pw12 (pos_weight=12) with defect_dice=0.756876, normal_fpr=0.000000, threshold=0.80, min_area=24.

## Source Paths And Hard-Normal Notes

### pw6

- Config: configs/stage2_p0_a40_e50_bs48_pw6.yaml
- Source output: outputs/experiments/p0_a40_e50_bs48_pw6_20260509
- OOF files: pw6/stage2/oof_global_postprocess.json, pw6/stage2/oof_global_postprocess_search.csv, pw6/stage2/oof_per_image.csv
- Fold logs: pw6/logs/fold{0..3}_stage2.log
- Hard-normal cap warnings observed: 141

### pw8

- Config: configs/stage2_p0_a40_e50_bs48_pw8.yaml
- Source output: outputs/experiments/p0_a40_e50_bs48_pw8_20260509
- OOF files: pw8/stage2/oof_global_postprocess.json, pw8/stage2/oof_global_postprocess_search.csv, pw8/stage2/oof_per_image.csv
- Fold logs: pw8/logs/fold{0..3}_stage2.log
- Hard-normal cap warnings observed: 93

### pw12

- Config: configs/stage2_p0_a40_e50_bs48_pw12.yaml
- Source output: outputs/experiments/p0_a40_e50_bs48_pw12_20260509
- OOF files: pw12/stage2/oof_global_postprocess.json, pw12/stage2/oof_global_postprocess_search.csv, pw12/stage2/oof_per_image.csv
- Fold logs: pw12/logs/fold{0..3}_stage2.log
- Hard-normal cap warnings observed: 112

## Included Files

- configs/: copied Stage2 pos_weight configs
- scripts/: copied batch launch script
- logs/: copied main batch log
- pw6/, pw8/, pw12/: copied fold logs, fold validation metrics/history/search tables, and pooled OOF outputs

Original experiment directories were preserved; this folder is a copied review bundle.
