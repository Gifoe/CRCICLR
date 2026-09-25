# Frozen C-direction Actionability Oracle Audit

This seed-zero EEGNet audit tests whether native C-direction erasure utility
implies useful amplification, whether trial-specific strengths have oracle
headroom, and whether that action can be predicted without labels. It reuses
the exact frozen V3 final-refit checkpoint, projectors, means, C basis, and
P-mediated utility. It does not train or modify the EEGNet backbone or its
classifier. A small diagnostic action classifier is fit only if the
prelocked oracle-headroom gate is met.

The only EEG arrays loaded by this experiment come from `inner_train_subjects`
and `inner_val_subjects` (discovery), using the ordinary non-final loader.
`outer_dev_subjects` and all final-heldout arrays are excluded. The V3
checkpoint and geometry were fitted in the prior experiment on its full
non-final refit pool, which included outer-dev subjects; this provenance is
retained in the lock. The direction indices and train-only ranking refer to
the exact V3 refit C basis. Their utility values use the frozen V2.5
P-mediated erasure-utility estimand carried by V3, so direction indices from
V2.5's separate PCA basis are not treated as the same directions. Results are
development diagnostics, not independent confirmation.

Run on the configured server after setting the V1, V2, V2.5, and V3 runtime
paths:

```bash
python experiments/persist_eeg_cp_actionability_oracle_v1_seed0/code/run.py preflight
python experiments/persist_eeg_cp_actionability_oracle_v1_seed0/code/run.py lock
python experiments/persist_eeg_cp_actionability_oracle_v1_seed0/code/run_queue.py evaluate --workers 1
python experiments/persist_eeg_cp_actionability_oracle_v1_seed0/code/run.py aggregate
```

The lock freezes the alpha grid, tie rules, TRAIN-only top-three ranking,
random controls, subject-level bootstrap, decision thresholds, and router
features before any current-experiment EEG is read. Evaluation streams one
subject/session at a time and uses one GPU worker. Full trial-level response
curves are stored as per-cell compressed CSV files under `outputs/cells/`;
`RESPONSE_CURVES.csv` contains subject/session summaries. If a task's discovery
oracle BA headroom reaches the prelocked 1 pp gate, aggregation fits and
evaluates the label-free diagnostic router for that task; otherwise it records
`ROUTER_PREDICTION_SKIPPED_LOW_HEADROOM` and does not fit a router.

Every result is labelled `DEVELOPMENT_ONLY_NO_FINAL_HELDOUT`. Do not use this
audit to tune or report final-heldout performance, and do not treat any oracle
as deployable.
