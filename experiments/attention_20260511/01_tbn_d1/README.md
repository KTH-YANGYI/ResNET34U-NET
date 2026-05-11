# 01 Transformer Bottleneck D1

Purpose: test whether a single transformer bottleneck improves global context modeling at the lowest-resolution U-Net feature map.

The only intended difference from `00_baseline_resnet34_unet_pw12` is:

- `transformer_bottleneck_enable: true`
- `transformer_bottleneck_layers: 1`
- `stage1_load_strict: false`, so compatible Stage1 weights load and new transformer parameters initialize fresh

All data, Stage1 checkpoints, loss settings, hard-normal replay settings, and post-processing search settings remain fixed.
