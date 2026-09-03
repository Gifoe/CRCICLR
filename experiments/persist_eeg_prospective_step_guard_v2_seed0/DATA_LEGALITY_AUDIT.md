# Data legality audit

Only the frozen OpenBMI 54-subject and WBCIC 41-subject development cohorts were used. Outcome rows were evaluated only in the declared development role and never entered gradients, normalization, or task batches. WBCIC outer 10 and OpenBMI sealed/internal confirmation data were not opened.

{
  "datasets": {
    "OpenBMI": {
      "fold_role_counts": [
        {
          "discovery": 9,
          "model_fit": 34,
          "outcome": 11
        },
        {
          "discovery": 9,
          "model_fit": 34,
          "outcome": 11
        },
        {
          "discovery": 9,
          "model_fit": 34,
          "outcome": 11
        },
        {
          "discovery": 9,
          "model_fit": 34,
          "outcome": 11
        },
        {
          "discovery": 9,
          "model_fit": 35,
          "outcome": 10
        }
      ],
      "outer_subject_ids_present": false,
      "rows": 10800,
      "sessions": [
        1,
        2
      ],
      "subjects": 54
    },
    "WBCIC": {
      "fold_role_counts": [
        {
          "discovery": 8,
          "model_fit": 24,
          "outcome": 9
        },
        {
          "discovery": 8,
          "model_fit": 25,
          "outcome": 8
        },
        {
          "discovery": 8,
          "model_fit": 25,
          "outcome": 8
        },
        {
          "discovery": 8,
          "model_fit": 25,
          "outcome": 8
        },
        {
          "discovery": 9,
          "model_fit": 24,
          "outcome": 8
        }
      ],
      "outer_subject_ids_present": false,
      "rows": 24591,
      "sessions": [
        0,
        1,
        2
      ],
      "subjects": 41
    }
  },
  "outcome_used_for_training": false,
  "pass": true,
  "sealed_outer_opened": false,
  "seed": 0
}
