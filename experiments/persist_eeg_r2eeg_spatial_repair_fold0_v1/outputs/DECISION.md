# SpatialRepair fold-0 decision

Terminal: `R2EEG_SPATIAL_REPAIR_CATASTROPHIC_FAIL_STOP`

R2EEG-v1 channel permutation: max logit 9.059906e-06, mean logit 3.7573045e-07, max latent 1.513958e-05, prediction agreement 1.0000.

BA: EEGNet 0.7786; v1 0.5157; SpatialRepair 0.6471. Recovery +13.14 pp; EEGNet gap -13.14 pp.

Interpretation: `NO`. INNER_VAL was used only for selection; OUTER_DEV was evaluated after checkpoint freeze. No reserved holdout was accessed.

Next action: diagnose the carrier before PRD.
