# EEGNet P task-core specificity closure, seed 0

Frozen mechanism audit of canonical EEGNet / OpenBMI_MI folds 0–4 at `embedding_64d`.
The source is commit `8a8b708b30e19a20a83902d272fbe53fe8279f12` and its
`persist_eeg_eegnet_p_semantics_canonical_geometry_seed0_v1` experiment.

`P_support` is the source TRAIN-only persistence decision; `U_support` is
reconstructed independently for **every** atomic block from the source
TRAIN-only absolute and excess erasure-utility 95% CI lower limits.  `G=P∩U`
must reproduce the source Protected coordinates and geometry exactly before
any OUTER_DEVELOPMENT access.  An empty comparison is reported as
`NOT_ESTIMABLE`, not zero performance.

No neural model is trained or updated.  Control matching and decoder fitting
use TRAIN_GEOMETRY only; OUTER_DEVELOPMENT is evaluation-only.
CHECKPOINT_VALIDATION remains checkpoint-selection-exposed.  Formal
final-heldout EEG is never opened.

The run is staged and fail-closed: `reconstruct` completes and verifies all
five folds before any `evaluate` call.  The `aggregate` phase checks all fold
manifests before producing the compact report.  Runtime caches, checkpoints,
and raw EEG are not committed.

The completed five-fold audit is in `outputs/`, with byte-level hashes in
`outputs/HASH_INDEX.json`. The corrected V4 evaluation uses the source 64-D
ambient residual for `C_CURRENT`; all non-C family rows were independently
checked identical to the preserved V2 evaluation. The final interpretation is
`MIXED_OR_INCONCLUSIVE` and the sole next-model decision is
`MECHANISM_NOT_IDENTIFIED_YET`. Utility support lies wholly within persistence
in these folds, so the direct persistence-beyond-utility contrast is not
estimable. Protected exceeds top PCA on task relation but not frozen decoder
BA, and the random energy/covariance controls do not achieve G's energy. Thus
the predeclared joint specificity gate is not passed; this does not establish
that energy/covariance fully explains all observed task relation.

The original server retains immutable reconstruction, V2/V4 evaluation,
P_ALL controls, subject bootstrap, and interim V3 aggregation artifacts.
The transient failed duplicate reconstruction trigger and interim overstrong
negative attribution label are disclosed in `outputs/FINAL_REPORT.md`.
