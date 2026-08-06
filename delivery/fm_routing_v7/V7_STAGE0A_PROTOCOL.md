# V7 Stage-0A protocol

```json
{
  "datasets": [
    "hmc",
    "eegmmidb"
  ],
  "outer_folds": [
    0,
    1,
    2,
    3,
    4
  ],
  "head_seeds": [
    0,
    1,
    2,
    3,
    4
  ],
  "subject_bootstrap": {
    "repetitions": 5000,
    "seed": 20260810
  },
  "subject_shuffle_null": {
    "repetitions": 500,
    "seed": 20260811
  },
  "probe": {
    "family": "LayerNorm+Linear",
    "learning_rate": [
      0.0001,
      0.0003,
      0.001
    ],
    "weight_decay": [
      0.0,
      0.0001
    ],
    "max_epochs": 30,
    "early_stopping_patience": 5,
    "loss": "class-weighted cross entropy"
  },
  "primary_unit": "subject",
  "backbones_frozen": true
}
```
