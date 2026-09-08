# Frozen EEGNet + LiteBN fusion/headroom audit

Primary results use only hash-verified `selected_best.pt` models, V8_SEARCH subjects, and the frozen five-fold split. No model was trained, updated, calibrated, stacked, or gated.

| Metric | OpenBMI | WBCIC |
|---|---:|---:|
| EEGNet BA | 0.7591 | 0.7857 |
| LiteBN BA | 0.7862 | 0.7821 |
| LiteBN delta (pp) | +2.708 | -0.358 |
| LOGIT50 BA | 0.7985 | 0.7916 |
| LOGIT50 delta (pp) | +3.942 | +0.587 |
| PROB50 BA | 0.7985 | 0.7916 |
| PROB50 delta (pp) | +3.942 | +0.587 |
| Subject Oracle BA (non-deployable) | 0.7921 | 0.7979 |
| LOGIT50 median subject delta (pp) | +3.667 | +0.705 |
| LOGIT50 harm <= -1pp | 2.5% | 12.9% |
| LOGIT50 positive folds | 5 | 3 |
| Oracle delta over best single carrier (pp) | +0.592 | +1.217 |

OpenBMI LOGIT50 retention of LiteBN gain: 145.5%.

## Terminal

`LIMITED_CONDITIONAL_HEADROOM`

The subject carrier oracle and trial complementarity upper bound are explicitly non-deployable label-informed diagnostics. They are not fusion methods and do not establish a viable gate.

## Fixed decision checks

```json
{
  "fixed_fusion_crossdataset_promising": false,
  "strong_conditional_checks": {
    "broad_subjects_ge_1pp": 18,
    "eegnet_only_fraction": 0.0779707747449683,
    "litebn_only_fraction": 0.0743782740556934,
    "open_oracle_ge_litebn": true,
    "oracle_over_EEGNet_pp": 1.2167915716302824,
    "stable_preference": true,
    "wbcic_oracle_minus_logit_pp": 0.6295345932442631
  }
}
```
