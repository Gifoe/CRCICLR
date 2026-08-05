# V6 Stage-0 frozen protocol

The online blockwise protocol was frozen before reading Oracle or online-policy results.

```json
{
  "schema_version": "online-blockwise-v6-stage0-v2",
  "alpha": 0.1,
  "delta": 0.1,
  "bootstrap_repetitions": 5000,
  "bootstrap_seed": 20260806,
  "permutation_repetitions": 50,
  "permutation_seed": 20260807,
  "tps_thresholds": [
    0.5,
    0.5257894736842105,
    0.5515789473684211,
    0.5773684210526315,
    0.6031578947368421,
    0.6289473684210526,
    0.6547368421052632,
    0.6805263157894736,
    0.7063157894736842,
    0.7321052631578947,
    0.7578947368421052,
    0.7836842105263158,
    0.8094736842105263,
    0.8352631578947368,
    0.8610526315789473,
    0.8868421052631579,
    0.9126315789473685,
    0.9384210526315789,
    0.9642105263157894,
    0.99,
    1.0
  ],
  "tps_grid_hash": "d158fadeeb2e35d52f7c641e3974bd88d4a661a14a17b6a1aadc934e6507c0a2",
  "K": 20,
  "hmc": {
    "block_epochs": 60,
    "tail_min_epochs": 30,
    "non_overlapping": true,
    "cross_recording_blocks": false
  },
  "eegmmidb": {
    "one_original_task_run_per_block": true,
    "minimum_valid_predictions": 8,
    "merge_runs": false,
    "preserve_run_order": true,
    "exclude_non_task_runs": true
  },
  "minimum_valid_blocks_per_subject": 4,
  "datasets": [
    "hmc",
    "eegmmidb"
  ],
  "source_seeds": [
    0,
    1,
    2,
    3,
    4
  ],
  "outer_folds": [
    0,
    1,
    2,
    3,
    4
  ],
  "backbone": "frozen CBraMod",
  "protected": {
    "formal_calibration_opened": false,
    "internal_final_opened": false,
    "cap_opened": false,
    "full_method_entered": false
  }
}
```
