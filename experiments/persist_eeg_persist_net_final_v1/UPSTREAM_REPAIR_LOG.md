# Upstream repair log

No upstream scientific definition has been changed.  The experiment uses the
Signed-V3.1 stable SHA256 seed principle and the exact DDA centered-logit
formula.

## V0 engineering repair 1

- Symptom: the first fold-0 source-only selection stopped before training with
  `ValueError: output array is read-only` while combining Arrow-backed boolean
  masks.
- Repair: explicitly copy the subject mask before an in-place session-mask
  conjunction.
- Scientific impact: none.  Predicate, rows, split, labels, configuration, and
  random seeds are unchanged.
- Outcome access before repair: none; the failure occurred while constructing
  inner-source normalizer indices.
- Rerun scope: fold-0 V0 selection from the beginning.

## V0 engineering repair 2

- Symptom: static pre-outcome audit found that training helpers set the stable
  seed after callers had already constructed each model. Optimization and
  dataloader order were seeded, but constructor-time parameter initialization
  was not. Matched dual-path controls also used method-specific initialization
  seeds.
- Repair: every fresh source model is explicitly reset after applying its
  stable SHA256 fold/seed. Matched dual-path methods now share the same
  initialization and source minibatch order within each fold/seed. A preflight
  unit check requires exact state-dict equality for equal seeds and a parameter
  difference for unequal seeds.
- Scientific impact: removes uncontrolled initialization variance;
  architecture, losses, hyperparameters, splits, certificates, gates, and
  outcome definitions are unchanged.
- Outcome access before repair: none. Only fold-0 source-only selection had
  completed. The four concurrent source-only selections were stopped before
  writing selection artifacts.
- Rerun scope: all five source-only selections from the beginning, followed by
  all final development runs.

## Reporting repair 3

- Static audit found a string/integer subject-ID mismatch in the final
  `per_subject_results.csv` delta lookup. It affected only that exported delta
  column, not gate calculations. The lookup now canonicalizes keys to strings.
- Outcome access before repair: none.

## Baseline reconstruction repair 4

- Static pre-outcome audit found that when B1 selected F16, the independently
  reconstructed F8 B0 would have reused B1's selected epoch instead of F8's own
  source-inner-validation epoch.
- Repair: B0 now always uses the F8 candidate's own frozen source-only epoch;
  B1 uses the selected candidate's epoch as before.
- Scientific impact: prevents an avoidably weak B0 and makes the conservative
  strongest-baseline comparison valid. No outer outcome was accessed and no
  candidate or epoch was added.

## Matched target-adaptation repair 5

- Symptom: static pre-outcome audit found that matched dual-path methods used
  method-specific target-adaptation seeds. In particular, A6 PUD all-adapt and
  A10 PUD protected-freeze started from the same source checkpoint but saw
  different target-history minibatch orders.
- Repair: every dual method for the same `(fold, seed, subject)` now uses
  `stable_seed("paired-dual-adapt", fold, seed, subject)`. The seed is exported
  in both the adaptation and mechanism ledgers.
- Scientific impact: removes an avoidable stochastic confound from the direct
  freeze-versus-all-adapt and matched-control comparisons. No split, model,
  loss, optimizer, epoch, certificate, metric, or gate changed.
- Outcome access before repair: none. Five seed-0 jobs were stopped after
  source training; a complete scan of 24 runtime log files found zero
  `[outcome]` markers. No Session-2 outcome result had been produced or read.
- Rerun scope: all 15 final development runs from the beginning. The completed
  five source-only selection locks remain valid because this repair affects
  target adaptation only.

## Server orchestration repair 6

- Symptom: the first repaired five-way run was launched as a descendant of an
  interactive Windows OpenSSH session. When that SSH connection was reset,
  the OpenSSH job object terminated the orchestrator and all five Python
  children even though PowerShell `Start-Process` had used a hidden window.
- Repair: final runs are launched by Windows Task Scheduler using
  `code/run_scheduled_server.ps1`. The task has an independent lifetime and the
  wrapper records stdout, stderr, and a machine-readable completion state.
- Scientific impact: none. No model, split, seed, certificate, optimizer,
  metric, outcome, or gate changed.
- Outcome access before repair: none. At interruption, a scan found zero
  `[outcome]` markers, zero `DONE.json` files, and zero run-error bytes. Only
  source-training progress had been emitted.
- Rerun scope: all 15 final development runs from the beginning. The five
  source-only selection locks remain unchanged.

## Post-run orchestration repair 7

- Symptom: all 15 workers wrote valid `RUN_COMPLETE` markers and complete
  subject/mechanism tables with empty stderr, but the Task Scheduler wrapper
  stopped before finalization. PowerShell had serialized every redirected
  child process `ExitCode` as `null`; the expression `$null -ne 0` then falsely
  marked every worker as failed.
- Repair: `run_all_server.ps1` now calls `WaitForExit()` before reading the
  native exit code. If Windows still returns `null`, it permits a zero-code
  recovery only when the exact fold/seed `DONE.json` says `RUN_COMPLETE` and
  that worker's stderr is empty; otherwise it records `-999` and fails closed.
- Scientific impact: none. No model, data, split, seed, checkpoint, outcome,
  statistic, gate, or report definition changed. The already completed workers
  were not rerun. The unchanged frozen finalizer was invoked once after an
  explicit audit found 15/15 valid completion markers, 15/15 subject tables,
  15/15 mechanism tables, and zero worker-error bytes.
- Outcome access before repair: all frozen development outcomes had been
  produced, but no aggregate performance result had been read and no model
  change was made. This repair only restores deterministic post-run control
  flow and records its evidence.

## Reporting consistency repair 8

- Symptom: the headline primary FULL-versus-baseline CI and the FULL row in the
  main table used two separately seeded 10,000-draw Monte Carlo bootstraps of
  the same 40 subject-level differences. Both were valid, but their endpoints
  differed slightly and created an avoidable reporting inconsistency.
- Repair: the exported FULL table row now reuses the registered primary
  bootstrap object already used for G2 and the headline. Other exploratory
  method rows retain their deterministic method-specific streams.
- Scientific impact: none. Subject differences, point estimate, gate values,
  pass/fail decisions, model outputs, and the registered primary bootstrap are
  unchanged. This is a presentation-only deduplication after finalization.

## Reporting portability repair 9

- Symptom: independent local replay of the result-only finalizer reached the
  Markdown export and failed because pandas treats `tabulate` as an optional
  dependency. All JSON/CSV/statistical outputs had already completed.
- Repair: the fixed, small main table now uses an internal deterministic
  Markdown renderer. No numeric computation or artifact schema changed.
- Scientific impact: none. The repair only removes an unnecessary optional
  reporting dependency; the result-only finalizer was rerun without training.
