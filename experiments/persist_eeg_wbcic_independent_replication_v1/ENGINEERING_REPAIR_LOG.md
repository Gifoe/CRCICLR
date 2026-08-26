# Engineering repair log

All repairs below were completed before the first outcome-S3 evaluation:

1. Reused the historical cache builder but restricted it to explicit paths from the 41-subject provenance whitelist; no raw-root enumeration.
2. Added a streamed float16 consolidated memmap to avoid duplicating full arrays in RAM.
3. Corrected WBCIC identity-probe session indices to BIDS ses-0↔ses-1 (S1↔S2); the inherited 1↔2 indices would have produced an empty direction.
4. Restored the missing run-level `SOURCE_FREEZE_COMPLETE.json` call before outcome evaluation.
5. Made source normalizer and persistence-basis freezes resume-safe and hash-checked.
6. Removed an unused OpenBMI-only conformer class from the Phase-3 runtime surface; EEGNet is the sole executed backbone.
7. Added resumable per-configuration checkpoints, deterministic controls, preflight compilation/scope checks, and independent final validation.

No subject scope, fold, seed, lambda, method, direction rank, metric, or statistical gate was altered after outcome access.


## Post-outcome aggregation-only repair

8. The first final aggregation attempt failed while reading the trusted, run-generated embeddings.npz: outcome_subjects was serialized as a NumPy object array, while the reader used allow_pickle=False. The aggregation reader now permits the existing object-string field. No training, source freeze, outcome evaluation, metric, bootstrap definition, threshold, or scientific result was changed. Aggregation was rerun from all 15 frozen run artifacts and independent validation passed.
