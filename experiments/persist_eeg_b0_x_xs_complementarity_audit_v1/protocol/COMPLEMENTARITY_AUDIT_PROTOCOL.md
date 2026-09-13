# Complementarity audit protocol

- Experiment type: analysis only.
- New model training, fitting, threshold selection, routing, or ensembling: prohibited.
- Data scope: development outer subjects and their historical evaluation session only.
- Final heldout, internal heldout, sealed test, and final-test result artifacts: prohibited.
- Pairing key: `task, seed, fold, subject_id, session, trial_id`.
- A paired cell is admissible only when both checkpoint files match their recorded SHA256 and use the same historical normalizer.
- Replay tolerance for subject BA, macro-F1, and accuracy: `1e-8` (hard ceiling `1e-6`).
- Missing exact checkpoint: `CHECKPOINT_UNAVAILABLE`; no retraining or substitution.
- Trial/label/normalizer mismatch: `PAIR_ALIGNMENT_FAIL`.
- Replay mismatch: `PROTOCOL_FAIL_FOR_CELL`, excluded from every downstream statistic.
- Aggregation: subject-equal. Raw trial counts/rates remain explicitly descriptive.
- Bootstrap: 10,000 paired resamples; unit is the real subject.
- Oracle: label-informed analytical upper bound only, never deployable performance.
- Practical follow-up threshold: at least `1.0 pp` subject-equal oracle headroom.

Replay uses one fixed deterministic numerical mode for every cell: full FP32, TF32 disabled, cuDNN benchmarking disabled, and deterministic cuDNN enabled. Any historical subject metric that does not reproduce under this fixed mode is marked `PROTOCOL_FAIL_FOR_CELL`; the entire fold cell is excluded rather than selecting numerical kernels post hoc.

The seed-0 union oracle uses the B0/X cell from the X experiment as the common baseline and the exact seed-0 XS prediction produced with the same normalizer. This is the only provenance-valid three-model comparison; pairwise B0-versus-XS results continue to use the B0 checkpoints frozen by the XS development experiment.
