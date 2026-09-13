# Post-gate C0 OpenBMI scope

The locked controlled-v2 WBCIC continuation gate did not pass. At the user's
explicit request, C0 DualCE seed 0 is nevertheless run on OpenBMI-MI,
OpenBMI-ERP, and OpenBMI-SSVEP as a post-gate exploratory analysis.

These runs do not retroactively authorize Phase 2, do not alter the frozen
WBCIC decision, and do not promote C0 to the original C1 candidate. They reuse
the exact historical LiteBN, canonical five-fold split, normalizers, training
exposure, task-specific supervised CE, checkpoint selection, and controlled
two-stream RNG/BatchNorm implementation. The already-open internal-heldout
cohort remains development/model-selection data. No new sealed test is used.
