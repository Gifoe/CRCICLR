# Cross-backbone CSGD v1

This experiment evaluates frozen selected checkpoints with their original
train-only channelwise normalizers and actual classification heads.  The
primary comparison uses seed 0 and five folds.  A secondary estimate is made
only for model-task combinations with complete seeds 0/1/2 checkpoint
matrices.

OpenBMI uses the fixed 14-subject internal-heldout diagnostic cohort at S1 and
S2.  WBCIC uses the fixed 10-subject true-outer cohort at S0, S1, and S2.
Folds are first averaged within each biological subject; confidence intervals
use 20,000 biological-subject bootstrap draws.

No training, finetuning, calibration, target adaptation, BN update,
normalization refit, or evaluation-label-based selection is permitted.

Runtime session/cell JSON and logs live outside git under
`D:\nips-temp\TotalP\P1\crossbackbone_csgd_runtime`.

TFFormer is retained in the requested audit matrix but has no exact checkpoint
identity in the repository.  Existing TCFormer artifacts are not aliased.
SGN is evaluated only for complete model-task matrices; its incomplete ERP and
SSVEP matrices remain explicitly reported as missing.
