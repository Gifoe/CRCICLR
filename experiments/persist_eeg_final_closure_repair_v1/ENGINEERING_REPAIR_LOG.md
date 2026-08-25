# Engineering repair log

The log is append-only after the repair protocol is frozen. Scientific thresholds, folds, seeds, targets, and terminal states are not edited in response to results.

## Authorized cache path resolution

The first server preflight failed before reading any EEG because the current Git worktree contains frozen run locks/checkpoints but not the ignored authorized cache. The implementation now resolves `PERSIST_DATA_EXPERIMENT` first and otherwise derives the historical experiment root from the frozen fold-0/seed-0 normalizer path in `RUN_LOCK.json`. It still requires the exact named V8_SEARCH metadata and raw-signal cache, and the 40-subject/8000-row guard remains unchanged. No split, subject, label, or result was inspected to make this repair.

## Adapted-result authoritative join

The first Phase-A aggregation attempt produced zero joined subjects and stopped before writing a result. Inspection of frozen method counts showed that `source_only_raw.csv` contains only the four source-only dual methods, while all 120 `PUD_AFTER_ADAPT` rows are in `adapted_authoritative_raw.csv`. The repair now reads that authoritative adapted table and adds it to the frozen-input hash audit. No statistic, threshold, or model output was changed.

## Hierarchical-bootstrap implementation acceleration

The first valid hierarchical-bootstrap run was interrupted because the initial implementation repeatedly allocated pandas subsets inside every nested draw. The implementation now pre-indexes the identical fold/run/direction/subject cells as NumPy arrays and computes the same 5,000 draws with the same seed and hierarchy. Statistical units, resampling probabilities, draw count, source scores, outcomes, and terminal rules are unchanged.

## Parallel-run protocol write guard

After the first Matched-TaskOnly candidate completed successfully, Phase B was authorized to run independent fold/seed jobs concurrently. The static `PHASE_B_MATCHED_PROTOCOL.md` writer now returns when the already-created identical report exists, avoiding a shared temporary-file race. Per-run training inputs, seeds, caches, selections, and outputs remain isolated by fold/seed.

A bounded server scheduler waits for all three fold-0 pilot seeds, fails on any traceback, and then runs at most three independent remaining fold/seed processes. This changes wall-clock scheduling only; each job's frozen initialization, minibatch order, source rows, selection, and result files are unchanged.
