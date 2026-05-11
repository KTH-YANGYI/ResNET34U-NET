# 00 Baseline ResNet34-U-Net PW12

Purpose: reproduce the best stable Stage2 baseline before adding attention modules.

This experiment uses the same Stage1 checkpoints and Stage2 training recipe as the attention variants. It exists inside the same experiment set so architecture deltas can be compared against a contemporaneous baseline rather than against an older run with different bookkeeping.

Expected outputs after running:

- `logs/fold{0..3}_stage2_train.log`
- `logs/oof_postprocess.log`
- `logs/holdout_ensemble.log`
- `results/stage2/fold{0..3}/`
- `results/stage2/oof_global_postprocess.json`
- `results/stage2/holdout_ensemble/holdout_metrics.json`
