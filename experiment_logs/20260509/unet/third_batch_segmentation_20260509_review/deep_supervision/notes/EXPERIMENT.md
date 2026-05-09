# deep_supervision

Purpose: deep supervision auxiliary segmentation heads on the pos_weight=12 baseline.

This is a third-batch experiment. It keeps the best pos_weight=12 baseline and
changes one training mechanism so the result can be compared directly against
the completed pos_weight ablation.

- Config: configs/stage2_p0_a40_e50_bs48_pw12_deepsup.yaml
- Runner: scripts/run_p0_a40_stage2_e50_bs48.sh
- Output root: outputs/experiments/p0_a40_e50_bs48_pw12_deepsup_20260509
- Fold logs: outputs/experiments/p0_a40_e50_bs48_pw12_deepsup_20260509/logs/fold{0..3}_stage2.log
- Pooled OOF log: outputs/experiments/p0_a40_e50_bs48_pw12_deepsup_20260509/logs/oof_postprocess.log
- Pooled OOF result: outputs/experiments/p0_a40_e50_bs48_pw12_deepsup_20260509/stage2/oof_global_postprocess.json
