# PERSIST-EEG Stage-0B — seed-0 CPU report

## Execution status

{
  "P0 baseline reproduction": "COMPLETE",
  "P0 random erasure": "COMPLETE",
  "P0 PCA erasure": "AUDITED_FROM_LEGACY",
  "P0 erasure efficacy": "COMPLETE",
  "P0 rank32 audit": "COMPLETE",
  "P1 Long": "COMPLETE",
  "P1 Medium": "COMPLETE",
  "P1 utility matrix": "COMPLETE",
  "P1 bootstrap": "COMPLETE",
  "P1 paradigm control": "DIAGNOSTIC_ONLY"
}

## Utility matrix (mean held-out-subject BA)

intervention | UL_only | UM_only | erase_UL | erase_UL+UM | erase_UM | pca_same_rank | random_same_rank | raw | residualized
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
erp | 0.545633 | 0.556553 | 0.649087 | 0.641172 | 0.648560 | 0.639460 | 0.649997 | 0.650099 | 0.650426
mi | 0.719291 | 0.708855 | 0.688218 | 0.671655 | 0.692218 | 0.580400 | 0.737806 | 0.738064 | 0.738182
ssvep | 0.859936 | 0.905864 | 0.908591 | 0.893791 | 0.904818 | 0.825173 | 0.924970 | 0.924845 | 0.926518

## Random same-rank control
{
  "erp": {
    "mean": -0.00010145013344554799,
    "std": 0.001830472875184535,
    "median": -5.509641873269189e-05
  },
  "mi": {
    "mean": -0.0002577272727272799,
    "std": 0.0069601686775541925,
    "median": 0.00022727272727274261
  },
  "ssvep": {
    "mean": 0.00012445454545452318,
    "std": 0.0030183283549670226,
    "median": 0.0
  }
}

## PCA same-rank control
PCA values are included in the matrix and are sourced from the frozen, outer-fold-local Stage-0 audit.

## Decisions
- `SEED0_SUPPORTS_SELECTIVE_PERSISTENCE_UTILITY`
- Medium: `MEDIUM_SUPPORTED`

## Limitations
- OpenBMI is offline/train-only as audited in Stage-0; no online/test labels are fabricated.
- PCA baseline and frozen P0 random/PCA diagnostics are preserved as legacy audit inputs; new Stage-0B utility probes are fold-local and refit.
- This is a seed-0 provisional result; no GO_PERSIST_UTILITY and no multi-seed inference.
- Medium evidence is task-heterogeneous: ERP is near chance (mean AUROC 0.532); the overall MEDIUM_SUPPORTED label is driven by MI/SSVEP and should not be generalized to ERP.
- Medium evidence is task-heterogeneous: ERP is near chance (mean AUROC 0.532); the overall MEDIUM_SUPPORTED label is driven by MI/SSVEP and should not be generalized to ERP.
- Medium evidence is task-heterogeneous: ERP is near chance (mean AUROC 0.532); the overall MEDIUM_SUPPORTED label is driven by MI/SSVEP and should not be generalized to ERP.
- Medium evidence is task-heterogeneous: ERP is near chance (mean AUROC 0.532); the overall MEDIUM_SUPPORTED label is driven by MI/SSVEP and should not be generalized to ERP.
- Medium evidence is task-heterogeneous: ERP is near chance (mean AUROC 0.532); the overall MEDIUM_SUPPORTED label is driven by MI/SSVEP and should not be generalized to ERP.