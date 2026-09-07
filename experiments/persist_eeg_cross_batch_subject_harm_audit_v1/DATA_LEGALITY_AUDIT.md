# Data legality audit

Only frozen source/refit biological subjects from OpenBMI and WBCIC were used. The session inventory and fold-level source/discovery role counts below come from the preflight runner. Outcome-role trials were not materialized for the audit; the `outcome_not_used` field records the role size that remained untouched. WBCIC outer-10 and OpenBMI sealed/confirmation resources were not opened. Seed 0 only; seed 1 and seed 2 were not run.

```json
{
  "OpenBMI_sealed_opened": false,
  "WBCIC_outer_opened": false,
  "batch_rule": "m_per_class=min(16,floor(min_available_per_class/5)); no replacement",
  "datasets": {
    "OpenBMI": {
      "fold_roles": [
        {
          "discovery": 9,
          "fold": 0,
          "model_fit": 34,
          "outcome_not_used": 11
        },
        {
          "discovery": 9,
          "fold": 1,
          "model_fit": 34,
          "outcome_not_used": 11
        },
        {
          "discovery": 9,
          "fold": 2,
          "model_fit": 34,
          "outcome_not_used": 11
        },
        {
          "discovery": 9,
          "fold": 3,
          "model_fit": 34,
          "outcome_not_used": 11
        },
        {
          "discovery": 9,
          "fold": 4,
          "model_fit": 35,
          "outcome_not_used": 10
        }
      ],
      "outer_subject_ids_present": false,
      "rows": 10800,
      "sessions": [
        1,
        2
      ],
      "source_only": true,
      "subjects": 54
    },
    "WBCIC": {
      "fold_roles": [
        {
          "discovery": 8,
          "fold": 0,
          "model_fit": 24,
          "outcome_not_used": 9
        },
        {
          "discovery": 8,
          "fold": 1,
          "model_fit": 25,
          "outcome_not_used": 8
        },
        {
          "discovery": 8,
          "fold": 2,
          "model_fit": 25,
          "outcome_not_used": 8
        },
        {
          "discovery": 8,
          "fold": 3,
          "model_fit": 25,
          "outcome_not_used": 8
        },
        {
          "discovery": 9,
          "fold": 4,
          "model_fit": 24,
          "outcome_not_used": 8
        }
      ],
      "outer_subject_ids_present": false,
      "rows": 24591,
      "sessions": [
        0,
        1,
        2
      ],
      "source_only": true,
      "subjects": 41
    }
  },
  "m_per_class": {
    "OpenBMI": [
      16
    ],
    "WBCIC": [
      16
    ]
  },
  "outcome_used": false,
  "schema": "PERSIST_CROSS_BATCH_DATA_LEGALITY_V1",
  "sealed_outer_opened": false,
  "seed": 0,
  "seed1_run": false,
  "seed2_run": false,
  "source_subjects_only": true
}
```
