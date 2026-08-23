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
