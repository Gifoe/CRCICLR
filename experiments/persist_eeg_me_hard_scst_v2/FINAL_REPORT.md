# Final report

{
  "branch": "codex/persist-eeg-me-hard-scst-v2",
  "terminal": "ME_HARD_SCST_NOT_SUPPORTED",
  "immutable_v1_terminal": "SCST_UTILITY_NOT_SUPPORTED_IN_NEAR_ADMISSIBLE_SPACE",
  "v1_code_map_status": "recovered; five artifact-backed methods reproduced; ShuffleSameClass historical value had no code/artifact",
  "v1_reproduction_pass": true,
  "selected_source_recipe": {
    "OpenBMI_delta_BA": 0.0002499999999999,
    "WBCIC_delta_BA": 0.0001797138047137,
    "bank_stability_min": 1.0,
    "coverage_min": 0.9639092242630264,
    "hardness_CI95_L_min": 0.0056348209828818,
    "lambda_H": 1.0,
    "minimum_development_delta": 0.0001797138047137,
    "q": 0.5,
    "scope": "A",
    "semantic_pass_min": 0.9591555909833556
  },
  "bank_factorization": {
    "median_norm_b": 0.6090346771690816,
    "median_norm_c": 0.5811327831521907,
    "median_main_effect_energy_fraction": 0.5521486132051316,
    "median_eta": 0.5
  },
  "source_coverage_mean": 0.9562787093836966,
  "source_median_valid": 24.0,
  "source_hardness_gap_mean": 0.007619679389112292,
  "discovery": {
    "bootstrap_draws": 10000,
    "comparisons": [
      {
        "CI95_L": -0.0010975609756097597,
        "CI95_U": 0.0013817032109715245,
        "comparison": "ME-HardSCST-ERM",
        "delta_BA": 0.00012154060934549811,
        "positive_folds": 4
      },
      {
        "CI95_L": -0.0008947195532561244,
        "CI95_U": 0.001624794694916661,
        "comparison": "ME-HardSCST-Mixup",
        "delta_BA": 0.0002841422353617566,
        "positive_folds": 4
      },
      {
        "CI95_L": -0.001018877802414384,
        "CI95_U": 0.0013817032109715085,
        "comparison": "ME-HardSCST-V1-RandomTransport",
        "delta_BA": 0.0001215406093454954,
        "positive_folds": 3
      },
      {
        "CI95_L": -0.0002439024390243769,
        "CI95_U": 0.00024390243902438233,
        "comparison": "ME-HardSCST-Dynamic-ClassConditional-Uniform-NoKL",
        "delta_BA": 0.0,
        "positive_folds": 2
      },
      {
        "CI95_L": -0.0002032520325203326,
        "CI95_U": 0.00020325203252031637,
        "comparison": "ME-HardSCST-Factorized-Uniform-NoKL",
        "delta_BA": -5.415722071342227e-18,
        "positive_folds": 2
      },
      {
        "CI95_L": 0.0,
        "CI95_U": 0.0,
        "comparison": "ME-HardSCST-Factorized-HardRandom",
        "delta_BA": 0.0,
        "positive_folds": 0
      }
    ],
    "diagnostic_gate": true,
    "discovery_supported": false,
    "outer_or_sealed_opened": false,
    "terminal_if_stop": "ME_HARD_SCST_NOT_SUPPORTED"
  },
  "confirmation": null,
  "outer_resource_status": "NOT_OPENED"
}
