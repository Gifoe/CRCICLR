# Final feasibility audit

```json
{
  "branch": "codex/persist-eeg-exp4-headroom-mechanism-audit-v1",
  "execution_base_commit": "dda0674e8524483be48667ae58e95ebf0403c54e",
  "Clean_Strong_Generic_BA": 0.7727499999999999,
  "historical_83_775_legal_for_holdout": false,
  "Binary_Generic_NoAdapt_oracle_headroom": 0.003999999999999998,
  "Expanded_action_bank_oracle_BA": 0.781,
  "Expanded_oracle_delta_vs_Generic": 0.008250000000000002,
  "Best_fixed_single_action": "A3_Generic_75pct",
  "Best_fixed_single_action_BA": 0.77475,
  "Personalization_headroom": 0.006249999999999978,
  "subjects_ge_1pp_recoverable_gain": 22,
  "fold_positive_oracle_count": 5,
  "HEADROOM_SUPPORTED": false,
  "old_Exp4_D_identical_to_Exp3_D": false,
  "old_vs_exact_D_difference": "Exp4-V1 used the fraction of labels changed after erasing protected contribution; Exp3 used continuous RMS class-centered logit displacement under finite erasure. Flip discards all sub-threshold magnitude information.",
  "Dfinite_harm_AUROC": 0.4982078853046595,
  "Dfinite_optimal_action_AUROC": 0.625,
  "Dflip_harm_AUROC": 0.6200716845878136,
  "Dflip_optimal_action_AUROC": 0.5416666666666667,
  "identity_AUROC": 0.4551971326164875,
  "confidence_AUROC": 0.7275985663082437,
  "update_magnitude_AUROC": 0.6630824372759856,
  "P_only_AUROC": 0.6845878136200717,
  "P_plus_exact_D_AUROC": 0.6200716845878136,
  "P_plus_U_plus_exact_D_AUROC": 0.6451612903225807,
  "confidence_update_P_exact_D_AUROC": 0.5806451612903226,
  "exact_D_adds_beyond_confidence_update": false,
  "P_confidence_Pearson": 0.2204201289785812,
  "P_confidence_Spearman": 0.24202626641651034,
  "P_confidence_prediction_vectors_identical": false,
  "P_confidence_equal_AUROC_explanation": "coincidental metric equality",
  "current_repaired_P_confidence_Pearson": 0.018532903073925994,
  "current_repaired_P_confidence_Spearman": -0.042026266416510326,
  "certified_directions_before_cap_per_fold_seed": [
    {
      "fold": 0,
      "seed": 0,
      "rank_before_cap": 11
    },
    {
      "fold": 0,
      "seed": 1,
      "rank_before_cap": 7
    },
    {
      "fold": 0,
      "seed": 2,
      "rank_before_cap": 10
    },
    {
      "fold": 1,
      "seed": 0,
      "rank_before_cap": 10
    },
    {
      "fold": 1,
      "seed": 1,
      "rank_before_cap": 5
    },
    {
      "fold": 1,
      "seed": 2,
      "rank_before_cap": 7
    },
    {
      "fold": 2,
      "seed": 0,
      "rank_before_cap": 10
    },
    {
      "fold": 2,
      "seed": 1,
      "rank_before_cap": 10
    },
    {
      "fold": 2,
      "seed": 2,
      "rank_before_cap": 9
    },
    {
      "fold": 3,
      "seed": 0,
      "rank_before_cap": 10
    },
    {
      "fold": 3,
      "seed": 1,
      "rank_before_cap": 8
    },
    {
      "fold": 3,
      "seed": 2,
      "rank_before_cap": 10
    },
    {
      "fold": 4,
      "seed": 0,
      "rank_before_cap": 8
    },
    {
      "fold": 4,
      "seed": 1,
      "rank_before_cap": 8
    },
    {
      "fold": 4,
      "seed": 2,
      "rank_before_cap": 8
    }
  ],
  "old_rank8_characterization": "CAP_SATURATION_AFTER_OLD_WEAK_P_U_FLIP_QUALIFICATION",
  "MECHANISM_SUPPORTED": false,
  "Phase_B_authorized": false,
  "selector_development": "NOT_REACHED",
  "internal_holdout_accessed": false,
  "holdout_materialized": false,
  "previous_terminal_state_unchanged": "EXP4_PERSIST_GUARD_NOT_SUPPORTED",
  "previous_gate_correction": "PREVIOUS_GATE_REPORTING_INCONSISTENCY: the old G label implied mechanism-risk superiority, but its code compared guard BA; PERSIST risk AUROC 0.613 was below confidence 0.728.",
  "Dfinite_numeric_validation": {
    "source_code_path": "experiments/persist_eeg_dda_v1/code/persist_dda_v1.py",
    "source_function": "centered_logit_sq + subject_decision_metrics",
    "definition": "sqrt(mean(sum((delta_logits-mean_class(delta_logits))^2, class)))",
    "binary_margin_equivalent": "sqrt(mean(delta_margin^2)/2)",
    "numeric_exact": 0.7125202900831432,
    "numeric_analytic": 0.7125202900831432,
    "absolute_error": 0.0,
    "archived_Exp3_report": "D:\\nips-temp\\TotalP\\P1\\CRCICLR_EXP3_DECISION_GROUNDING_CLOSURE_V1\\experiments\\persist_eeg_exp3_decision_grounding_closure_v1\\EXP3_FINAL_REPORT.json",
    "archived_Exp3_values": {
      "decision_protected_mean": 0.9982230109222217,
      "decision_control_mean": 0.2467850870938559,
      "M0_RMSE": 0.04597839942134,
      "MI_RMSE": 0.0457441624640147,
      "MD_RMSE": 0.0314928431971294,
      "MID_RMSE": 0.0315332767866679,
      "M0_minus_MD": 0.014791624471360496,
      "MD_beats_MI_positive_runs": 6
    },
    "archived_values_match_frozen_reference": true,
    "validated": true
  },
  "terminal_state": "EXP4_STOP_INSUFFICIENT_ACTION_HEADROOM",
  "strongest_justified_claim": "The tested five-action family falls below the predeclared oracle-headroom gate, and exact Exp3 decision dependence adds no prospective harm information beyond the strongest confidence/update control; constructive Exp4 development must stop.",
  "strongest_unsupported_claim": "PERSIST provides a validated prospective action selector, improves the sealed holdout, or outperforms confidence/update controls."
}
```
