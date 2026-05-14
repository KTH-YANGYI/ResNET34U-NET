# 03 Skip Attention Gate D4+D3

Purpose: test whether decoder skip attention gates suppress irrelevant encoder features while preserving useful spatial detail.

The only intended difference from `00_baseline_resnet34_unet_pw12` is:

- `skip_attention_enable: true`
- `skip_attention_levels: ["d4", "d3"]`
- `skip_attention_gamma_init: 0.0`

The zero gamma initialization makes the gate start close to an identity residual path, reducing the chance that the new gate suppresses skip features before training has evidence.
