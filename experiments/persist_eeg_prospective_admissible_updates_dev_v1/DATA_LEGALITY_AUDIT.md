# Data legality audit

PASS: only frozen OpenBMI and WBCIC development roles were used. WBCIC uses the 41 authorized development subjects; the sealed outer 10 and any OpenBMI sealed/internal confirmation cohort were not opened. No outcome row enters training, normalization or epoch selection; outcome rows are used only as the declared development evaluation.

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
