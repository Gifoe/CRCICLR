# Scientific context audit

This experiment is a pre-registered transfer-risk audit, not a method search.
It uses only OpenBMI MI and WBCIC MI, canonical folds 0--4, EEGNet, and seed 0.

## Reuse and non-reuse

- `persist_eeg_canonical_eegnet_baseline` is reused only for the frozen EEGNet
  architecture and optimizer/epoch-selection semantics.  Its refit checkpoints
  are not used to define pseudo-unseen difficulty because those checkpoints have
  seen discovery subjects.
- Signed V3.1 is reused for the scientific definitions (cross-session spectrum,
  SHA256 sampling, persistence null, utility assignment, EPS/threshold logic,
  and matched-rank controls).  Its concrete spectra and assignments are not
  reused.
- Shared Geometry V1.2 is reused for subject/session centering, binary class
  directions and cosine geometry.  Its fold artifacts are not reused.
- The WBCIC independent-replication lock supplies the development subject pool
  and role audit only; its historical basis is not silently mixed with the
  current A-only representation.

## Fold provenance

OpenBMI Signed V3.1/Shared Geometry used historical 3-fold/seed matrices, while
this audit uses the canonical five-fold role file.  WBCIC historical persistence
artifacts also use a different pipeline and role naming.  Therefore no old fold
number is paired with a current fold number.  Every current fold rebuilds its
persistence basis from its own model-fit A-only embeddings.

## Data boundary

For each fold A is model-fit and B is discovery.  C is the canonical outcome
role and is never indexed, enumerated, or label-read by this runner.  OpenBMI
descriptor trials come from B S1+S2 and its query is residual B S2 trials;
WBCIC descriptor trials come from B S1+S2 and its query is B S3.  The WBCIC
outer-10 lock and any OpenBMI outer/sealed cohort are not opened.

The preflight phase writes all geometry and descriptor locks before any B query
performance is evaluated.  Only the outcome phase, after
`PRE_OUTCOME_PROTOCOL_LOCK.json`, reads query labels.
