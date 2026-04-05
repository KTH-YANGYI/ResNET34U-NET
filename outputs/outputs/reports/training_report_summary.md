# Training Report Summary

- Outputs root: `C:\Users\18046\Desktop\master\masterthesis\UNET_two_stage\outputs\outputs`
- Stage2 history does not include validation loss, so the Stage2 plots show train loss and validation metrics only.

## Stage1 Summary

| fold | epochs_ran | best_epoch | best_patch_dice_all | best_patch_dice_pos_only | best_positive_patch_recall | best_negative_patch_fpr | replay_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold0 | 32 | 20 | 0.751327 | 0.613681 | 0.928571 | 0.101695 | 71 |
| fold1 | 44 | 32 | 0.776427 | 0.535104 | 0.866667 | 0.007463 | 53 |
| fold2 | 52 | 40 | 0.762186 | 0.637972 | 1.000000 | 0.090000 | 67 |
| fold3 | 40 | 28 | 0.735532 | 0.579880 | 0.966667 | 0.103448 | 63 |

## Stage2 Summary

| fold | epochs_ran | best_epoch | defect_dice | defect_iou | defect_image_recall | normal_fpr | threshold | min_area | stage2_score | raw_defect_dice | raw_normal_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold0 | 16 | 4 | 0.570298 | 0.452748 | 0.809524 | 0.097335 | 0.540000 | 48 | 0.570298 | 0.578754 | 0.155272 |
| fold1 | 13 | 1 | 0.548851 | 0.417259 | 0.950000 | 0.065854 | 0.820000 | 32 | 0.548851 | 0.504386 | 0.131707 |
| fold2 | 13 | 1 | 0.683931 | 0.529533 | 1.000000 | 0.044118 | 0.880000 | 48 | 0.683931 | 0.527926 | 0.156863 |
| fold3 | 33 | 21 | 0.572971 | 0.431020 | 0.950000 | 0.099640 | 0.420000 | 32 | 0.572971 | 0.551632 | 0.097239 |
