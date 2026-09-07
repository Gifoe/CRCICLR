# Source repair rationale

The single allowed repair was chosen before outcome access from the source-only gate: {
  "competence_fail": true,
  "diversity_fail": false,
  "pre_repair_report": {
    "OpenBMI": {
      "min_expert_delta_BA": -0.023333333333333317,
      "mean_expert_delta_BA": -0.010611111111111089,
      "mean_pairwise_disagreement": 0.031074074074074077,
      "mean_rme_delta_BA": -0.003777777777777769,
      "mean_rme_delta_NLL": 0.0002700342449649673,
      "competence_pass": false,
      "diversity_pass": true,
      "rme_BA_pass": false
    },
    "WBCIC": {
      "min_expert_delta_BA": -0.22687499999999994,
      "mean_expert_delta_BA": -0.030166903409090933,
      "mean_pairwise_disagreement": 0.11676401224892605,
      "mean_rme_delta_BA": 0.0015555555555555101,
      "mean_rme_delta_NLL": 0.009220330255626897,
      "competence_pass": false,
      "diversity_pass": true,
      "rme_BA_pass": true
    },
    "cross_dataset_nll_pass": false
  },
  "new_recipe": {
    "beta_risk": 0.25,
    "lambda_kd": 0.5
  }
}
