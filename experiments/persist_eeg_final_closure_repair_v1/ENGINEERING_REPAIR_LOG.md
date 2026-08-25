# Engineering repair log

The log is append-only after the repair protocol is frozen. Scientific thresholds, folds, seeds, targets, and terminal states are not edited in response to results.

## Authorized cache path resolution

The first server preflight failed before reading any EEG because the current Git worktree contains frozen run locks/checkpoints but not the ignored authorized cache. The implementation now resolves `PERSIST_DATA_EXPERIMENT` first and otherwise derives the historical experiment root from the frozen fold-0/seed-0 normalizer path in `RUN_LOCK.json`. It still requires the exact named V8_SEARCH metadata and raw-signal cache, and the 40-subject/8000-row guard remains unchanged. No split, subject, label, or result was inspected to make this repair.
