# Validation hold: future-session error prediction

The 20 frozen cells and aggregate completed, but the original prediction analysis is not cleared for a prospective scientific claim. A separate correction was completed and independently audited; the original invalid output remains preserved.

In `code/run_coupling.py`, `stage_cell` builds `outer_feature_pairs` with `pair_indices(a_outer, oy, ...)`. `pair_indices` uses the recipient's and donor's true class labels to select cross-label pairs. The resulting `outer_coupling_features` are passed to `error_predictability` for the future-session outer-development subjects. Thus the reported `PC_ERROR_PREDICTABILITY.csv` and REPORT future-session AUROC use outer labels in feature construction, not only for evaluation. They cannot support a prospective unseen-subject error-prediction claim or success criterion C as written.

The core 2x2 cross-label mechanism pairs require true classes for retrospective evaluation and need not be changed. The defect is specifically reusing those label-conditioned pairs to create features for a prospective error predictor. The user authorized a separately provenanced label-free correction. Its proposed donor rule uses frozen native predicted classes and nearest activation norm within biological subject and session. It does not alter the frozen protocol or original outputs. Because the rule was specified after the outer results were viewed, even a valid corrected analysis is exploratory and cannot be represented as a preregistered success-criterion C test.

Do not change the frozen protocol or overwrite the original cell, stage-cache, log, or aggregate evidence. The exact engineering correction and its separate provenance must be reviewed before any corrected predictability result is treated as valid. No Phase 2 mechanism-closure work may rely on the current predictability claim.

Observed frozen protocol SHA256: `3ddb5006bf7159b00a0c80fed31856849ccdaa69ecf5060b0161c8673327e15c`.

Observed implementation SHA256: `4aab984969f4ba43128ed738a38ac138d94ee1c9b1252662af91b0c44e3f8eee`.

Final-heldout accessed: NO.

## Resolution and limits

All 20 separate label-free repair cells completed with the original geometric baseline unchanged. The corrected aggregate and compact publication audit completed successfully; see `outputs_compact/PUBLISH_VALIDATION.json` for original and repaired file hashes, row counts, per-cell verification, and control counts. The corrected ALL_LAYER minus BASE native-error AUROC deltas are approximately +0.001 (EEGNet MI), +0.002 (EEGNet SSVEP), -0.013 (EEGConformer MI), and -0.001 (EEGConformer SSVEP). None establishes the proposed stable positive prediction result; criterion C is not supported. These numbers are exploratory because the donor rule was chosen after inspecting the invalid outer result, and because same-subject/session unlabeled batch access is required.

The frozen retrospective mechanism analyses were not changed. Under the locked criterion A, 13 model/task/layer summaries have a biological-subject CI lower bound above zero and at least four positive folds. This is a mechanism progression gate, subject to the report's multiple-comparison caveat, not a validation of the original predictor.
