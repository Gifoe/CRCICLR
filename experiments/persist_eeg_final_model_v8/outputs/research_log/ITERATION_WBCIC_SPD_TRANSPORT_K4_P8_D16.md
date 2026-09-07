# WBCIC_SPD_TRANSPORT_K4_P8_D16

Structural hypothesis: shrinkage spectral SPD geometry, history-only whitening, and learned spatial metrics provide complementary cross-session structure absent from deep embeddings.

```json
{
  "benchmark": "WBCIC_S1S2_to_S3_authorized_development",
  "family_id": "WBCIC_S1S2_to_S3_authorized_development__SPD_TRANSPORT_K4_P8_D16",
  "subjects": 12,
  "baseline_method": "B_STRONG_MATCHED_V7",
  "baseline_BA": 0.8137499999999999,
  "strongest_single_candidate": "SPD_TRANSPORT_K4_P8_D16__E0_RESIDUAL50",
  "strongest_single_candidate_BA": 0.8137499999999999,
  "mean_expert_BA": 0.8135416666666666,
  "subject_oracle_BA": 0.8183333333333334,
  "oracle_headroom_pp": 0.45833333333333376,
  "subjects_rescued_ge_2pp_fraction": 0.0,
  "subjects_rescued_ge_5pp_fraction": 0.0,
  "positive_fold_fraction": 1.0,
  "mean_pairwise_correctness_correlation": 0.9835194651763618,
  "mean_pairwise_correctness_disagreement": 0.005,
  "oracle_usage": {
    "B_STRONG_MATCHED_V7": 7,
    "SPD_TRANSPORT_K4_P8_D16__E0_RESIDUAL50": 2,
    "SPD_TRANSPORT_K4_P8_D16__E1_RESIDUAL50": 1,
    "SPD_TRANSPORT_K4_P8_D16__E2_RESIDUAL50": 0,
    "SPD_TRANSPORT_K4_P8_D16__E3_RESIDUAL50": 2
  },
  "oracle_assignment_entropy": 1.1187433359857524,
  "headroom_state": "V8_HEADROOM_WEAK",
  "outcome_labels_used_for_headroom_only": true,
  "used_to_train_selector": false,
  "internal_holdout_used": false,
  "OUTER_TEST_USED": false,
  "experts": 4,
  "spatial_filters": 8,
  "reduced_dimension": 16,
  "epochs": 250,
  "tau": 0.08,
  "lambda_mean": 0.35,
  "folds": [
    0,
    1
  ],
  "baseline_source_method": "ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD",
  "training_objective": "history-whitened shrinkage SPD class prototypes with query-trained spatial metric coverage",
  "deployment_transform": "locked strong anchor plus half learned SPD residual",
  "covariance_cache": {
    "reused": false,
    "rows": 18591,
    "channels": 58,
    "sampling_rate_hz": 250,
    "band_hz": [
      8.0,
      30.0
    ],
    "shrinkage": 0.05,
    "internal_holdout_rows": 0,
    "path": "D:\\nips-temp\\TotalP\\P1\\CRCICLR_V8_HEADROOM_FIRST\\experiments\\persist_eeg_final_model_v8\\outputs\\cache\\WBCIC_V8_SEARCH_SPECTRAL_COV_FLOAT16.npy",
    "OUTER_TEST_USED": false
  }
}
```
