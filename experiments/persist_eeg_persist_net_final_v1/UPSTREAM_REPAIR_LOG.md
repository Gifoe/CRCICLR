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
