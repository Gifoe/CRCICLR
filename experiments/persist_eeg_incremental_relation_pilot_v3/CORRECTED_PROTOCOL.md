# Corrected incremental relation residual pilot

{
  "all_existing_cache_retained": true,
  "architecture": {
    "B1": "64->16->2 zero-init output",
    "B2": "[64 latent; 2 prototype evidence]->16->2 zero-init output",
    "B3": "DeepSets phi 4->8->8, subject-balanced mean, rho [64;8]->12->2 zero-init output"
  },
  "backbone": "canonical EEGNet fold0 seed0 SUBJECT_BALANCED_ERM frozen",
  "created_at_utc": "2026-09-05T08:58:56Z",
  "data_scope": {
    "OpenBMI_sealed_holdout_opened": false,
    "WBCIC_outer_10_opened": false,
    "source_only_before_lock": true
  },
  "datasets": [
    "OpenBMI",
    "WBCIC"
  ],
  "final_form": "frozen SB-ERM logits plus trainable residual logits",
  "fold": 0,
  "hyperparameters": {
    "batch_size": 512,
    "loss": "subject/class-balanced cross entropy",
    "lr": 0.001,
    "max_epochs": 30,
    "optimizer": "Adam",
    "patience": 5,
    "weight_decay": 0.0001
  },
  "methods": [
    "SUBJECT_BALANCED_ERM",
    "GENERIC_RESIDUAL",
    "GENERIC_PROTOTYPE_RESIDUAL",
    "CROSS_SUBJECT_SESSION_RELATION_RESIDUAL"
  ],
  "outcome_aware_changes": false,
  "representation": "full 64-d latent",
  "schema": "PERSIST_EEG_CORRECTED_INCREMENTAL_RELATION_AMENDMENT_V3",
  "seed": 0,
  "status": "ACTIVE"
}
