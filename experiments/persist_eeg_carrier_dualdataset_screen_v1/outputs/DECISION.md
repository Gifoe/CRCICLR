# Carrier dual-dataset fold0 screen

All candidates were fixed before outer development evaluation. No holdout or WBCIC true outer data were loaded.

| model | openbmi_BA | openbmi_delta_pp | openbmi_macroF1 | openbmi_subject_median_delta_pp | wbcic_BA | wbcic_delta_pp | wbcic_macroF1 | wbcic_subject_median_delta_pp | min_gain_pp | mean_gain_pp | parameter_count | terminal_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Compact | 0.8229 | 2.4286 | 0.8201 | 0.0000 | 0.7600 | -3.2727 | 0.7517 | -2.0000 | -3.2727 | -0.4221 | 182094.0000 | MIXED |
| LiteBN | 0.8300 | 3.1429 | 0.8263 | 1.5000 | 0.7582 | -3.4545 | 0.7451 | -1.0000 | -3.4545 | -0.1558 | 47978.0000 | MIXED |
| LiteGN | 0.8157 | 1.7143 | 0.8114 | -0.5000 | 0.7677 | -2.5000 | 0.7668 | -3.0000 | -2.5000 | -0.3929 | 47978.0000 | MIXED |
| MSResidual | 0.8007 | 0.2143 | 0.7967 | 1.5000 | 0.7805 | -1.2273 | 0.7798 | -1.0000 | -1.2273 | -0.5065 | 16922.0000 | MIXED |
| MSResidualRMS | 0.7871 | -1.1429 | 0.7808 | -0.5000 | 0.7814 | -1.1364 | 0.7807 | -1.0000 | -1.1429 | -1.1396 | 16922.0000 | NEGATIVE |

Terminal gates are descriptive architecture-screen gates, not confirmation claims.

## Protocol lock (re-stated)

```json
{
  "datasets": [
    "OpenBMI",
    "WBCIC"
  ],
  "epochs": 60,
  "fold": 0,
  "gradient_clip": 5.0,
  "lr": 0.0003,
  "optimizer": "AdamW",
  "order": "train all models in both datasets before any outer-dev evaluation",
  "seed": 0,
  "selection": "future-session inner validation BA, epoch 10..60",
  "split_sha256": "050703ca8676ae43f236691ed37d58d4ed7ed97b83f449a36f04d93af9ebcd14",
  "weight_decay": 0.0005
}
```

## Holdout isolation audit

```json
{
  "V8_INTERNAL_HOLDOUT_loaded": false,
  "WBCIC_true_outer_loaded": false,
  "scope": "V8_SEARCH / historical Stage-1 development fold0 only"
}
```

## Pre-evaluation tests

```json
{
  "V8_INTERNAL_HOLDOUT_not_loaded": true,
  "WBCIC_true_outer_not_loaded": true,
  "all_architectures_defined_before_outer_evaluation": true,
  "lite_bn_gn_only_normalizer_difference": true,
  "only_fold0": true,
  "outer_dev_indices_absent_from_training_episodes": {
    "OpenBMI": true,
    "WBCIC": true
  },
  "residual_final_layer_zero_initialized": true
}
```

## Model specs (parameter counts)

```json
{
  "Compact": {
    "class": "exact historical CompactEncoder",
    "params_58": 181806,
    "params_62": 182094,
    "source_sha256": "7f453a7cb4b85e66dfadb28c130777567bea87dcab7ac3173bb752a88581a409"
  },
  "EEGNet": {
    "class": "canonical EEGNet",
    "params_58": 34098,
    "params_62": 34162
  },
  "LiteBN": {
    "class": "CompactLite BN",
    "params_58": 47786,
    "params_62": 47978
  },
  "LiteGN": {
    "class": "CompactLite GN",
    "params_58": 47786,
    "params_62": 47978
  },
  "MSResidual": {
    "class": "frozen EEGNet + MSResidualBranch",
    "params_58": 16730,
    "params_62": 16922
  }
}
```
