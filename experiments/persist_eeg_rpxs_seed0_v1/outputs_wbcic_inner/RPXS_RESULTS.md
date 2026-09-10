# RPXS seed0 canonical-inner result

- Source branch/commit: `codex/persist-eeg-rpxs-seed0-v1` / `d67133ae58e660368a6db5025f061c71abf98f8d`
- Architecture: `LiteBN_RPXS` = XS architecture initialized from LiteBN-X checkpoints.
- Optimizer: AdamW; channel LR `0.0001`, X-initialized LR `3e-05`, weight decay `0.0005`.
- Residual-path dropout: batch-level `p_gate_on=0.5`; L2-SP `0.001`; beta_gate `0.0001`.
- Selection: canonical inner-val BA with step0 X fallback; no outer/test access.
- Step0 exact X replay: **5/5 passed**.

| Task | mean X BA | mean RPXS BA | mean ΔBA pp | median ΔBA pp | +/0/- folds | mean ΔF1 |
|---|---:|---:|---:|---:|---:|---:|
| WBCIC_MI | 0.7980 | 0.7982 | +0.0200 | +0.0000 | 1/4/0 | -0.0000 |

WBCIC positive folds: **1/5**; mean ΔBA: **+0.0200 pp**; median ΔBA: **+0.0000 pp**.

## Diagnostics

- Gate-on/off diagnostic is in `RPXS_DIAGNOSTICS.csv` and was not used for selection.
- Full train trajectory is in `RPXS_TRAINING_TRAJECTORY.csv`.

RPXS_WBCIC_ONLY_COMPLETE

`OUTER_DEVELOPMENT_ACCESSED = NO`
`FINAL_HELDOUT_ACCESSED = NO`
