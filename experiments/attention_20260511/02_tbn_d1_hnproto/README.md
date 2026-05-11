# 02 Transformer Bottleneck + Hard-Normal Prototype Attention

Purpose: test whether hard-normal prototype memory adds useful negative context beyond the transformer bottleneck.

This experiment first builds a fold-specific prototype bank from the fixed Stage1 checkpoint, then trains Stage2 with prototype cross-attention.

The intended architecture difference from `01_tbn_d1` is:

- `prototype_attention_enable: true`
- Fold-specific prototype banks under `results/prototype_banks/fold{fold}/prototype_bank.pt`

The paired comparison `02` versus `01` isolates the added value of prototype attention, while `02` versus `00` measures the combined effect.
