# pw12_fp003

Purpose: normal_fp_loss_weight=0.03 on the pos_weight=12 Stage2 baseline.

This is the remaining second-batch experiment after hard-normal replay was confirmed stable.
It keeps pos_weight=12, threshold search up to 0.95, and adds a small normal false-positive
loss term to test whether normal false positives can be reduced without hurting defect recall.

- Config: configs/stage2_p0_a40_e50_bs48_pw12_fp003.yaml
- Runner: scripts/run_p0_a40_stage2_e50_bs48.sh
- Output root: outputs/experiments/p0_a40_e50_bs48_pw12_fp003_20260509
- Fold logs: outputs/experiments/p0_a40_e50_bs48_pw12_fp003_20260509/logs/fold{0..3}_stage2.log
- Pooled OOF log: outputs/experiments/p0_a40_e50_bs48_pw12_fp003_20260509/logs/oof_postprocess.log
- Pooled OOF result: outputs/experiments/p0_a40_e50_bs48_pw12_fp003_20260509/stage2/oof_global_postprocess.json
