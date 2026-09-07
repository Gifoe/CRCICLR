# PERSIST-EEG Shared Geometry Audit V1.2

This directory contains the development-only Shared Geometry audit that follows
the reproducibility repair in Signed Audit V3.1. It never accesses outer-test
data and never trains PERSIST-USE.

## Frozen protocol

- Runs: folds `0,1,2` × seeds `0,1`.
- Input basis: one persisted Signed-V3.1 canonical spectrum per run.
- Protected coordinates: frozen Signed-V3.1 assignments; no downstream basis
  reconstruction or block remapping.
- Data: TRAIN plus development validation only.
- Geometry/transfer cap: deterministic SHA256-selected `32` trials per
  subject/session/event group. Unique subject is the statistical unit; session
  directions are averaged within subject before Gates A/C bootstrap.
- Controls: 100 matched non-Protected coordinate controls, 100 orthogonal
  controls, same-rank PCA, non-Protected coordinates, and label permutation.
- Bootstrap: 10,000 draws; no trial-level pseudoreplication.

## Result

`results_v1_2/SHARED_GEOMETRY_FINAL_REPORT.json` records:

`SHARED_GEOMETRY_V1_2_PASS`

All MI Gates A–F pass:

- Gate A cross-subject contrast: mean `+0.3367`, hierarchical 95% CI
  `[0.2394, 0.4365]`.
- Gate B true LOSO prototype transfer: mean `+0.1581 BA`, CI
  `[0.1215, 0.1971]`.
- Gate C cross-subject × cross-session direction: mean `+0.3395`, CI
  `[0.2471, 0.4367]`.
- Gate D actual centroid margin: mean `+0.4037`, CI `[0.3158, 0.4991]`.
- Gate E utility link: geometry coefficient `+0.02446`, bootstrap CI
  `[0.01365, 0.04226]`, pooled Spearman `+0.55585`.
- Gate F: protected-only perturbation invariance, no outer-test, no spectrum
  rebuild.

The final report uses `204` subject-level values for each primary hierarchical
gate (34 subjects × 6 runs). In fold 2/seed 1, ERP and SSVEP have no frozen
Protected assignment in Signed V3.1; they are recorded as
`NO_PROTECTED_ASSIGNMENT` and do not affect the MI primary decision.

## Provenance states

- Historical Shared Geometry V1: `SHARED_GEOMETRY_AUDIT_V1_INVALID`.
- V1.1: `SHARED_GEOMETRY_V1_1_BLOCKED_BY_UPSTREAM_PROVENANCE` because the old
  Signed-V3 sampler used process-dependent `hash()` and did not persist indices;
  this is not a scientific mechanism failure.
- Signed V3.1 prerequisite: `PERSISTENCE_UTILITY_ASSIGNMENT_REPRODUCIBLE`.

The canonical Signed-V3.1 code and artifacts are under
`experiments/persist_eeg_p4_signed_v3_1/`. Raw EEG and caches are not included
in this repository export.
