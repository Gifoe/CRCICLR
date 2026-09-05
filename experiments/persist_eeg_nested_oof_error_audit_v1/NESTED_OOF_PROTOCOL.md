# Nested OOF protocol

{
  "schema": "PERSIST_EEG_NESTED_OOF_PROTOCOL_V1",
  "created_at_utc": "2026-09-05T10:45:17Z",
  "seed": 0,
  "datasets": [
    "OpenBMI",
    "WBCIC"
  ],
  "outer_K": 5,
  "inner_K": 4,
  "backbone": "canonical SUBJECT_BALANCED_ERM EEGNet recipe",
  "source_role": "role[0].model_fit",
  "methods": [
    "B0_OOF_ERM",
    "B1_GENERIC_OOF",
    "B2_GENERIC_PROTOTYPE_OOF",
    "B3_CROSS_SESSION_RELATION_OOF"
  ],
  "raw_latent_cross_model_mix": false,
  "outcome_labels_read": false,
  "triage_rule": "after two outer folds per dataset stop only if B3 deltas and B3-vs-B2 are all <=0"
}
