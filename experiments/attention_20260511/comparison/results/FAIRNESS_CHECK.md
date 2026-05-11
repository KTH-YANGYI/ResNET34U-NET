# Fairness Check

Status: PASS

## Fixed Settings

| key | value |
| --- | --- |
| seed | OK: 42 |
| image_size | OK: 640 |
| samples_path | OK: "manifests/samples.csv" |
| batch_size | OK: 48 |
| epochs | OK: 50 |
| encoder_lr | OK: 2e-05 |
| decoder_lr | OK: 0.0001 |
| weight_decay | OK: 0.0001 |
| bce_weight | OK: 0.5 |
| dice_weight | OK: 0.5 |
| pos_weight | OK: 12.0 |
| normal_fp_loss_weight | OK: 0.0 |
| normal_fp_topk_ratio | OK: 0.1 |
| amp | OK: true |
| early_stop_patience | OK: 16 |
| early_stop_min_delta | OK: 0.0 |
| lr_factor | OK: 0.5 |
| lr_patience | OK: 4 |
| min_lr | OK: 1e-07 |
| num_workers | OK: 8 |
| pretrained | OK: false |
| device | OK: "cuda" |
| auto_evaluate_after_train | OK: true |
| threshold | OK: 0.5 |
| train_eval_threshold | OK: 0.5 |
| train_eval_min_area | OK: 0 |
| use_imagenet_normalize | OK: true |
| augment_enable | OK: true |
| augment_hflip_p | OK: 0.5 |
| augment_vflip_p | OK: 0.5 |
| augment_rotate_deg | OK: 10 |
| augment_brightness | OK: 0.1 |
| augment_contrast | OK: 0.1 |
| augment_gamma | OK: 0.0 |
| augment_noise_std | OK: 0.015 |
| augment_blur_p | OK: 0.0 |
| target_normal_fpr | OK: 0.1 |
| lambda_fpr_penalty | OK: 2.0 |
| threshold_grid_start | OK: 0.1 |
| threshold_grid_end | OK: 0.95 |
| threshold_grid_step | OK: 0.02 |
| min_area_grid | OK: [0, 8, 16, 24, 32, 48] |
| random_normal_k_factor | OK: 1.0 |
| use_hard_normal_replay | OK: true |
| stage2_hard_normal_ratio | OK: 0.4 |
| hard_normal_max_repeats_per_epoch | OK: 2 |
| hard_normal_warmup_epochs | OK: 2 |
| hard_normal_refresh_every | OK: 2 |
| hard_normal_pool_factor | OK: 3.0 |
| deep_supervision_enable | OK: false |
| boundary_aux_enable | OK: false |
| stage1_checkpoint_template | OK: "outputs/experiments/p0_a40_20260508/stage1/fold{fold}/best_stage1.pt" |

## Allowed Architecture Differences

| experiment | key | expected | actual |
| --- | --- | --- | --- |
| 00_baseline_resnet34_unet_pw12 | transformer_bottleneck_enable | false | OK: false |
| 00_baseline_resnet34_unet_pw12 | prototype_attention_enable | false | OK: false |
| 00_baseline_resnet34_unet_pw12 | skip_attention_enable | false | OK: false |
| 00_baseline_resnet34_unet_pw12 | stage1_load_strict | true | OK: true |
| 01_tbn_d1 | transformer_bottleneck_enable | true | OK: true |
| 01_tbn_d1 | transformer_bottleneck_layers | 1 | OK: 1 |
| 01_tbn_d1 | transformer_bottleneck_heads | 8 | OK: 8 |
| 01_tbn_d1 | transformer_bottleneck_dropout | 0.1 | OK: 0.1 |
| 01_tbn_d1 | prototype_attention_enable | false | OK: false |
| 01_tbn_d1 | skip_attention_enable | false | OK: false |
| 01_tbn_d1 | stage1_load_strict | false | OK: false |
| 02_tbn_d1_hnproto | transformer_bottleneck_enable | true | OK: true |
| 02_tbn_d1_hnproto | transformer_bottleneck_layers | 1 | OK: 1 |
| 02_tbn_d1_hnproto | transformer_bottleneck_heads | 8 | OK: 8 |
| 02_tbn_d1_hnproto | transformer_bottleneck_dropout | 0.1 | OK: 0.1 |
| 02_tbn_d1_hnproto | prototype_attention_enable | true | OK: true |
| 02_tbn_d1_hnproto | prototype_attention_heads | 8 | OK: 8 |
| 02_tbn_d1_hnproto | prototype_attention_dropout | 0.1 | OK: 0.1 |
| 02_tbn_d1_hnproto | prototype_pos_max | 128 | OK: 128 |
| 02_tbn_d1_hnproto | prototype_neg_max | 128 | OK: 128 |
| 02_tbn_d1_hnproto | prototype_l2_normalize | true | OK: true |
| 02_tbn_d1_hnproto | prototype_batch_size | 64 | OK: 64 |
| 02_tbn_d1_hnproto | skip_attention_enable | false | OK: false |
| 02_tbn_d1_hnproto | stage1_load_strict | false | OK: false |
| 03_skipgate_d4d3 | transformer_bottleneck_enable | false | OK: false |
| 03_skipgate_d4d3 | prototype_attention_enable | false | OK: false |
| 03_skipgate_d4d3 | skip_attention_enable | true | OK: true |
| 03_skipgate_d4d3 | skip_attention_levels | ["d4", "d3"] | OK: ["d4", "d3"] |
| 03_skipgate_d4d3 | skip_attention_gamma_init | 0.0 | OK: 0.0 |
| 03_skipgate_d4d3 | stage1_load_strict | false | OK: false |
