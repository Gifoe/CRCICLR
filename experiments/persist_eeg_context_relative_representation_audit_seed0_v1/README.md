# Frozen recording-context representation audit

EEGNet / OpenBMI_MI / seed 0 / folds 0–4 only. This experiment tests a frozen, unlabeled same-recording reference; it does not train a neural model, modify P/C, or read formal final-heldout EEG.

The primary deployable comparison is FIRST-B context followed by disjoint outcome trials, with B, decoder regularization, and shrinkage selected using TRAIN_GEOMETRY only. OUTER_DEVELOPMENT is historically exposed and evaluation-only. CHECKPOINT_VALIDATION is inventory-only.

Important algebraic constraint: subtracting a single session reference from every trial leaves within-session class-difference vectors exactly unchanged. Consequently the stated Gate A cannot pass for REL_MEAN. We retain that predeclared gate, report it as an algebraic failure, and do not reinterpret a cosine or drift tie as improvement.

## Result

Across five folds (40 OUTER biological subjects), ABSOLUTE balanced accuracy was 0.738993 and REL_MEAN was 0.744592: a paired difference of +0.005599, with a biological-subject bootstrap 95% CI of [-0.008041, 0.018562]. Four folds were positive, but Gate B fails both the +0.010 threshold and positive-CI condition. Gate A fails by exact translation invariance; the measured class-relation cosine change was approximately zero. Mean session-within-subject variance *increased* by 0.095290 under REL_MEAN. Gates C, D and E pass under their operational definitions. The locked case label is `CURRENT_SESSION_REFERENCE_REQUIRED_BUT_GAIN_SMALL`, with mandated action `STOP_ARCHITECTURE_DIRECTION`; “required” in that label does **not** establish a general necessity claim, because the primary effect CI includes zero and the geometry premise failed. No final-heldout EEG was read.

The compact deliverables are in `outputs/`; the exact input checkpoint, split, normalizer, stage and source hashes are in `protocol/SOURCE_PROVENANCE.json` and `protocol/REPRESENTATION_STAGE_AUDIT.json`. `outputs/FINAL_REPORT.md` gives all twelve required answers and the fold table. The extra `FULL_SESSION_LOO_DIAGNOSTIC.csv` is transductive and is not used for the deployable result.

On the original server, the successful extraction entry was deployed as `run_v2.py` (matching the committed `code/run.py` SHA), TRAIN geometry as `geometry_v2.py` (matching `code/geometry.py`), localization as `localization_v1.py` (matching `code/localization.py`), and finalization as `finalize_v2.py` (matching `code/finalize.py`). Earlier operational attempts and filenames are retained in the server source/hash inventory; the fold-0 failed extraction is disclosed in `protocol/FAILURE_DISCLOSURES.md`. Runtime caches, EEG arrays, raw logs and checkpoints are deliberately excluded from Git.
