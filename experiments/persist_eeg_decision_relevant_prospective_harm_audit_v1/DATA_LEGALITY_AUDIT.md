# Data legality audit

Only frozen model-fit/discovery source/refit subjects and their legal fit sessions are used. B1--B4 and B_out are trial-disjoint. No development outcome trials, outcome labels, WBCIC outer-10, or OpenBMI sealed/confirmation trials are loaded or materialized.

```json
{
  "K": 4,
  "OpenBMI_sealed_opened": false,
  "WBCIC_outer_opened": false,
  "datasets": {
    "OpenBMI": {
      "fold_roles": [
        {
          "discovery_subjects": 9,
          "fold": 0,
          "model_fit_subjects": 34,
          "outcome_role_subjects_not_materialized": 11,
          "refit_rows": 8600,
          "source_subjects": 43
        },
        {
          "discovery_subjects": 9,
          "fold": 1,
          "model_fit_subjects": 34,
          "outcome_role_subjects_not_materialized": 11,
          "refit_rows": 8600,
          "source_subjects": 43
        },
        {
          "discovery_subjects": 9,
          "fold": 2,
          "model_fit_subjects": 34,
          "outcome_role_subjects_not_materialized": 11,
          "refit_rows": 8600,
          "source_subjects": 43
        },
        {
          "discovery_subjects": 9,
          "fold": 3,
          "model_fit_subjects": 34,
          "outcome_role_subjects_not_materialized": 11,
          "refit_rows": 8600,
          "source_subjects": 43
        },
        {
          "discovery_subjects": 9,
          "fold": 4,
          "model_fit_subjects": 35,
          "outcome_role_subjects_not_materialized": 10,
          "refit_rows": 8800,
          "source_subjects": 44
        }
      ],
      "observed_subjects": 54,
      "outer_subject_ids_present": false,
      "rows": 10800,
      "sessions": [
        1,
        2
      ],
      "subjects_in_frozen_development_pool": 54
    },
    "WBCIC": {
      "fold_roles": [
        {
          "discovery_subjects": 8,
          "fold": 0,
          "model_fit_subjects": 24,
          "outcome_role_subjects_not_materialized": 9,
          "refit_rows": 12796,
          "source_subjects": 32
        },
        {
          "discovery_subjects": 8,
          "fold": 1,
          "model_fit_subjects": 25,
          "outcome_role_subjects_not_materialized": 8,
          "refit_rows": 13196,
          "source_subjects": 33
        },
        {
          "discovery_subjects": 8,
          "fold": 2,
          "model_fit_subjects": 25,
          "outcome_role_subjects_not_materialized": 8,
          "refit_rows": 13197,
          "source_subjects": 33
        },
        {
          "discovery_subjects": 8,
          "fold": 3,
          "model_fit_subjects": 25,
          "outcome_role_subjects_not_materialized": 8,
          "refit_rows": 13198,
          "source_subjects": 33
        },
        {
          "discovery_subjects": 9,
          "fold": 4,
          "model_fit_subjects": 24,
          "outcome_role_subjects_not_materialized": 8,
          "refit_rows": 13197,
          "source_subjects": 33
        }
      ],
      "observed_subjects": 41,
      "outer_subject_ids_present": false,
      "rows": 24591,
      "sessions": [
        0,
        1,
        2
      ],
      "subjects_in_frozen_development_pool": 41
    }
  },
  "fit_sessions": {
    "OpenBMI": [
      1,
      2
    ],
    "WBCIC": [
      0,
      1
    ]
  },
  "m_per_class": 16,
  "outcome_data_materialized_before_lock": false,
  "outcome_index_created_before_lock": false,
  "outcome_labels_read_before_lock": false,
  "schema": "PERSIST_SSPG_DATA_LEGALITY_V1",
  "seed": 0,
  "seed1_run": false,
  "seed2_run": false,
  "source_only_training_subjects": true
}
```
