# Research lineage ledger

This is grouped by hypothesis family, not by branch. Exploratory and superseded rows are not confirmatory evidence.

| Family | Role | Terminal | Experiment path |
|---|---|---|---|
| A Persistence definition / P1 / P2 | AUTHORITATIVE_PRIMARY | `P2_PASS_MULTI_SEED_PERSISTENCE_UTILITY` | `experiments/persist_eeg_p3closure_p4; experiments/persist_eeg_persist_net_source_only_diagnostic_v1` |
| B P3 trajectory / compression | NEGATIVE_AUTHORITATIVE | `SELECTIVE_COMPRESSION_NOT_SUPPORTED` | `experiments/persist_eeg_p3closure_p4` |
| C PCA / high-variance relationship | AUTHORITATIVE_SUPPORTING | `PCA_OVERLAP_HIGH_BUT_NOT_IDENTITY` | `experiments/persist_eeg_p3closure_p4; experiments/persist_eeg_exp4_openbmi_final_v1` |
| D identity interpretation / matched identity causal audit | AUTHORITATIVE_SUPPORTING | `IDENTITY_INTERPRETATION_BOUNDED` | `experiments/persist_eeg_matched_identity_causal_v1_2; experiments/persist_eeg_final_closure_repair_v1` |
| E shared geometry / causal geometry | AUTHORITATIVE_SUPPORTING | `SHARED_GEOMETRY_V1_2_PASS` | `experiments/persist_eeg_p4_shared_geometry; experiments/persist_eeg_p4_signed_v3_1` |
| F PERSIST-PB / PERSIST-SI | NEGATIVE_AUTHORITATIVE | `P4_SELECTIVE_INVARIANCE_NOT_SUPPORTED` | `experiments/persist_eeg_p3closure_p4; experiments/persist_eeg_p4_selective_invariance` |
| G protected / nuisance / GRL | NEGATIVE_AUTHORITATIVE | `SAFETY_OR_ACTIONABILITY_GATE_NOT_SUPPORTED` | `experiments/persist_eeg_p4c_suppression_safety_validation_v1; experiments/persist_eeg_p4d_method_level_bridge_v1` |
| H PERSIST-Net | NEGATIVE_AUTHORITATIVE | `PERSIST_NET_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED` | `experiments/persist_eeg_persist_net_final_v1` |
| I PUD / PUD-Aux | NEGATIVE_AUTHORITATIVE | `PUD_AUX_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED` | `experiments/persist_eeg_final_failure_localization_and_aux_v1; experiments/persist_eeg_final_closure_repair_v1` |
| J SSPG / prospective gradient signal | AUTHORITATIVE_SUPPORTING | `PROSPECTIVE_SIGNAL_NOT_ACTIONABLE_OR_UNCERTAIN` | `experiments/persist_eeg_stable_subject_prospective_guard_seed0_v1; experiments/persist_eeg_cumulative_subject_decision_drift_audit_v1` |
| K transport / GeoSR / prototype / relation | EXPLORATORY_ONLY | `NO_STABLE_GENERAL_ACTIONABILITY_SIGNAL` | `experiments/persist_eeg_persistence_geometry_transfer_risk_audit_v1; experiments/persist_eeg_geosr_final_v1; experiments/persist_eeg_incremental_relation_pilot_v3` |
| L PDA / U-PDA / TEA / routing / selectors | NEGATIVE_AUTHORITATIVE | `UTILITY_OR_ROUTING_NOT_CERTIFIED` | `experiments/persist_eeg_prospective_action_policy_v2_1; experiments/persist_eeg_utility_certified_pda_final; experiments/persist_eeg_router` |
| M nested OOF / correction / utility gate | AUTHORITATIVE_SUPPORTING | `NO_CERTIFIED_PROSPECTIVE_UTILITY` | `experiments/persist_eeg_nested_oof_error_audit_v1; experiments/persist_eeg_prospective_utility_gate_v1` |
| N multi-backbone closure | NEGATIVE_AUTHORITATIVE | `FINAL_MULTIBACKBONE_FALSIFICATION_CLOSURE` | `experiments/persist_eeg_multibackbone_final_closure` |
| O SCST / competence-generality | NEGATIVE_AUTHORITATIVE | `SCST_UTILITY_NOT_SUPPORTED_IN_NEAR_ADMISSIBLE_SPACE` | `experiments/persist_eeg_scst_competence_generality_v1; experiments/persist_eeg_scst_utility_stage1` |
| P V4-V8 / headroom | AUTHORITATIVE_SUPPORTING | `V8_SCIENTIFIC_EXHAUSTION_PHASE_A_HEADROOM` | `experiments/persist_eeg_final_model_v4; experiments/persist_eeg_final_model_v5; experiments/persist_eeg_final_model_v6; experiments/persist_eeg_final_model_v7; experiments/persist_eeg_final_model_v8` |
| Q GroupDRO / MLDG / Fishr / style extrapolation | NEGATIVE_AUTHORITATIVE | `GENERIC_SUBJECT_DG_NOT_CERTIFIED` | `experiments/persist_eeg_route_b_foundation_screen_v1; experiments/persist_eeg_mldg_robustness_confirm_v1` |
| R MLDG robustness / per-seed early stopping | NEGATIVE_AUTHORITATIVE | `MLDG_PER_SEED_EARLY_STOP_PARTIAL_ONLY` | `experiments/persist_eeg_mldg_perseed_earlystop_check_v1; experiments/persist_eeg_mldg_robustness_confirm_v1` |
| S TSEG / trajectory stability | NEGATIVE_AUTHORITATIVE | `TWO_TRAJECTORY_GAIN_EXPLAINED_OR_NOT_GENERAL` | `experiments/persist_eeg_tseg_session_fairness_v1` |
| T additional PERSIST experiments | AUTHORITATIVE_SUPPORTING | `EMPIRICAL_CHAIN_INCOMPLETE` | `experiments/persist_eeg_exp3_decision_grounding_closure_v1; experiments/persist_eeg_dda_v1; experiments/persist_eeg_external_actionability_v1` |

## Explicit supersession decisions

- Shared Geometry V1 and V1.1 are superseded by V1.2; the earlier sampler/provenance was invalid or blocked.
- Early TSEG results with session exposure confounding are superseded by the session-fair closure.
- Fixed-epoch MLDG interpretations are superseded by the per-seed early-stop audit.
- Initial unpaired PUD-Aux numbers are not causal confirmation; matched repair is the usable development comparison.
- Historical best numbers are never cherry-picked as confirmatory evidence.
