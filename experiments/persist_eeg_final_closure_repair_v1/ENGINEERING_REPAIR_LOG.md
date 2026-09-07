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

## Post-run aggregation guard

A separate server-side finalizer waits for all 15 valid `RUN_COMPLETE.json` markers and aborts if the bounded scheduler reports a traceback. It then invokes the already-frozen `matched_aux.py --aggregate` entry point exactly once. This prevents connection loss from leaving completed training unaggregated and does not alter any scientific computation.

## Phase B matched repair

Added strict inner_train-only teacher/certificate construction, post-selection outer rebuild, class-centered RMS-matched random targets, exact initialization/minibatch SHA audits, resumable fold/seed caches, and a newly trained Matched-TaskOnly control. Outcome Session-2 labels were evaluated only after each run's selection artifact was frozen.

## Independent final artifact validation

After aggregation, a separate validator rechecks row/method/subject/seed counts, duplicate keys, fold membership, all 15 completion markers, inner and outer initialization SHA matching, nested subject separation, outcome-selection guards, frozen Phase-A hashes, restricted-access flags, gate/terminal consistency, and required final artifacts. It writes only `results/final_validation.json`; no metric or outcome is recomputed for model selection.
