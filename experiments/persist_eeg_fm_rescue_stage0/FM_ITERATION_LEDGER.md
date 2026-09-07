# FM iteration ledger

## V0 official-checkpoint full fine-tuning

- Diagnosis: official checkpoints and final 200-D representations load correctly; dataset adapters must repair only sampling, unit and channel-index requirements.
- Change: maximal legal channels, 200-Hz four-patch input, official checkpoint, full-model AdamW fine-tuning and a new two-class head.
- Evidence available: repository/checkpoint/input audits and source-validation only.
- Prediction: competent source validation without layer or outcome search.
- Outcome evidence inspected: NO.
- Keep/reject: pending the frozen source-validation search.

## V2 historical SCST gate-equivalence repair

- Diagnosis: a pre-primary static comparison against the hash-locked final SCST Repair-2 implementation found that the draft FM runner summarized same-subject cross-session residual cosine directly and used absolute affinity changes. The frozen historical protocol instead requires matched-minus-mismatched residual stability, relative target-affinity improvement, and SCST-minus-norm-random advantage, each aggregated at source-subject level with 10,000 subject bootstrap resamples and the applicable CI gates.
- Change: restored the historical matched-minus-mismatched stability effect; relative affinity and random-control effects; source-subject grouping across folds and seeds; independent-probe BA >= 0.55; FM task competence; stability, subject-fidelity, class-fidelity, and manifold sub-gates. Added the missing SCAA zero-harm/zero-coverage guard and fixed the report-only WBCIC scale description to x20,000 with the frozen +/-250 uV bound.
- Evidence available before change: historical SCST protocol locks and code, source-validation training logs only. No held-out FM task BA, D>I consequence, S2/S3 utility, FM SCST, or sealed-resource result was generated or inspected.
- Prediction: prevents false transport authorization and makes FM SCST numerically comparable to final Repair-2; no directional outcome prediction.
- Result: pending primary run.
- Keep/reject: KEEP and include in the pre-outcome protocol hash lock.

## V4 compact-report provenance repair

- Diagnosis: the draft unified table contained rounded placeholder specialist task BAs and the competence Markdown would otherwise remain a pre-freeze placeholder after outcome completion.
- Change: replaced placeholders with exact historical ERM/SCAA anchor means from the committed OpenBMI and WBCIC result tables, made finalization write the actual FM task BA, frozen threshold, pass/fail, and margin, and made historical utility-transferability cells follow the committed Spearman CI-lower > 0 evidence instead of a hand-coded value.
- Evidence available before change: committed historical specialist results and current source-validation logs only; no FM primary outcome was inspected.
- Prediction: reporting-only correction; no effect on any FM metric or terminal.
- Result: pending finalization.
- Keep/reject: KEEP and include in the pre-outcome protocol hash lock.

## V3 seed-grouping statistical repair

- Diagnosis: pre-primary static review found that the draft D>I runner used fold-by-seed as the leave-one-run-out and bootstrap group. That permits the same fold under other random seeds to remain in the regression training data and violates the explicit rule that seeds are not independent people.
- Change: all three seeds are now held out together by fold; the ridge comparison is leave-one-fold-out; the 10,000 hierarchical bootstrap resamples folds independently within dataset and synchronizes each sampled fold across the two FMs of that dataset.
- Evidence available before change: source-validation training logs only. No held-out FM task BA, D>I consequence, S2/S3 utility, FM SCST, or sealed-resource result was generated or inspected.
- Prediction: wider and more defensible D>I uncertainty; no directional outcome prediction.
- Result: pending primary run.
- Keep/reject: KEEP and include in the pre-outcome protocol hash lock.

## V5 SCAA grouped-evidence gate repair

- Diagnosis: pre-primary gate review found that the draft pooled SCAA sign gate checked only the 0.65 point estimate and did not enforce meaningful grouped evidence above 0.5. It also did not explicitly prevent a task-weak FM from authorizing constructive rescue.
- Change: pooled sign concordance is now formed per subject with both FM rows held together, uses 10,000 subject bootstrap resamples, and requires CI lower > 0.5 for a strong rescue. Strong rescue also requires both WBCIC FMs to pass the frozen task-competence threshold; an architecture-dependent label requires the individually positive FM to be competent.
- Evidence available before change: task prompt, frozen competence thresholds, and source-validation logs only. No held-out FM task BA, S2/S3 utility, or other primary outcome was generated or inspected.
- Prediction: prevents a correlated-row or task-weak false-positive rescue; no directional outcome prediction.
- Result: pending primary run.
- Keep/reject: KEEP and include in the pre-outcome protocol hash lock.

## V0 decision

The globally selected source-validation recipes and S1-only head recipes were retained. No layer, channel, outcome, S2 or S3 search occurred. The primary protocol is now frozen.

## V6 representation-cache serialization repair

- Diagnosis: the first primary invocation stopped before producing any task-performance, D>I, SCAA, or SCST result because pandas subject identifiers were cached as NumPy object arrays while the frozen loader correctly required `allow_pickle=False`.
- Change: normalize object-valued cache fields to fixed-width Unicode inside `save_rep`; discard only the three incomplete caches created by the failed invocation so they are deterministically regenerated. The loader remains `allow_pickle=False`.
- Evidence available before change: Python traceback, cache key/dtype inspection, and the already frozen source-validation/S1-only selections. No primary outcome file existed or was inspected.
- Prediction: identical numerical representations and labels, with subject IDs serialized safely; no metric or terminal can change except that primary computation can proceed.
- Result: pending repaired primary run.
- Keep/reject: KEEP as a pre-outcome engineering repair and refresh the protocol code hash before rerun.

## V7 scikit-learn API compatibility repair

- Diagnosis: after all representations were cached and task-performance output was written but not inspected, D>I stopped before its first fitted cell because the installed scikit-learn version no longer accepts the deprecated `multi_class="auto"` constructor argument.
- Change: remove only `multi_class="auto"` from the identity-probe `LogisticRegression`; current scikit-learn's default behavior is equivalent for this multiclass `lbfgs` fit. All `C`, solver, iterations, folds, seeds, features, labels, and outcome definitions remain frozen.
- Evidence available before change: exception type and installed API behavior. No task-BA values, D>I cell, S2/S3 utility, or SCST result was inspected; D>I/SCAA/SCST output files did not exist.
- Prediction: API compatibility only; no directional metric prediction.
- Result: pending repaired primary run.
- Keep/reject: KEEP as an engineering compatibility repair and refresh only the `run_primary.py` protocol hash before rerun.

## V8 native-crash bootstrap/resume repair

- Diagnosis: all 60 D>I fold-seed units and the complete cell, prediction, and four-setting summary tables were written, then CPython 3.11 crashed in `python311.dll` with `0xC0000005` before the D>I bootstrap JSON and before SCAA began. The three D tables were not numerically inspected.
- Change: pre-aggregate the already specified two-FM synchronized value for each `(dataset, fold)` and perform the same 10,000 hierarchical draws with NumPy indexing instead of repeated pandas boolean allocations. Add a resume path that requires exactly 480 cells, 1,920 regression predictions, four setting rows, and no duplicate keys before producing the unchanged statistic.
- Evidence available before change: file existence/row structure, execution log, Windows APPCRASH event, and source review. No D>I values, S2/S3 utility, or SCST result was inspected.
- Prediction: identical resampling estimand, seed, draw count, grouping, CI quantiles, and terminal; lower allocation pressure allows the fresh process to continue into SCAA/SCST.
- Result: pending repaired primary continuation.
- Keep/reject: KEEP as a scientifically equivalent engineering recovery; refresh the `run_primary.py` hash before continuation.

## V9 pandas-2 grouped SCAA bootstrap repair

- Diagnosis: all 30 SCAA fold-seed adaptation units and complete seed-level, subject-level, and per-FM summary tables were written, then pandas 2.x raised an internal `Index` context-manager `TypeError` during pooled subject resampling. SCST had not started, and SCAA values were not inspected.
- Change: align both FM tables once by the frozen common subject IDs, convert `Delta_S2`/`Delta_S3` to NumPy arrays, and apply one shared integer subject-resample vector to both FMs on every draw. Add a resume path requiring unique model-subject and model-fold-seed-subject keys, both FMs, and exactly three seed rows per subject row.
- Evidence available before change: traceback, output-file existence/row structure, and source review. No SCAA numeric result or SCST output was inspected.
- Prediction: exactly the same grouped-subject estimand, stable seed, 10,000 draws, correlation statistic, sign bootstrap, gates, and terminal without pandas `.loc` allocation/API dependence.
- Result: pending repaired primary continuation.
- Keep/reject: KEEP as a scientifically equivalent engineering compatibility repair; refresh the `run_primary.py` hash before continuation.

## V10 representation-stage resume isolation

- Diagnosis: the repaired SCAA NumPy bootstrap was entered only after a fresh process unnecessarily reloaded all 60 FM anchors; CPython then repeated the same native access violation before SCST. All representation caches and task tables had already completed deterministically.
- Change: reuse the task/representation stage only after validating exactly 60 unique `(dataset, model, fold, seed)` task rows, four unique summaries, and every one of the 210 expected role caches. Otherwise the original extraction path runs. This isolates downstream CPU statistics from cumulative PyTorch/FM native state.
- Evidence available before change: cache/task table existence and key counts, execution stage markers, and the repeated Windows APPCRASH code. No task, D>I, SCAA numeric values or SCST result was inspected.
- Prediction: no scientific value changes; the resumed process enters CPU-only SCAA statistics without loading 60 anchors and then proceeds to SCST.
- Result: pending isolated continuation.
- Keep/reject: KEEP as deterministic checkpoint/resume engineering; refresh the `run_primary.py` hash before continuation.

## V11 finalizer historical-table dependency repair

- Diagnosis: finalization wrote the frozen no-trigger decision and then stopped while drawing the admissibility matrix because `figures()` referenced the committed historical SCAA correlation table through a local variable that existed only in `make_table()`.
- Change: load the same committed `UTILITY_TRANSFER_CORRELATION.csv` inside `figures()` before constructing the historical utility-transferability cells.
- Evidence available before change: NameError traceback, committed historical table, and completed frozen FM results. The repair does not inspect or branch on any new outcome.
- Prediction: report/figure generation completes with the already frozen historical evidence; all FM metrics, gates, triggers, and terminals remain byte-for-byte unchanged.
- Result: pending finalization rerun.
- Keep/reject: KEEP as reporting-only dependency injection; refresh the `finalize.py` protocol hash.
