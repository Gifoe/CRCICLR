# Experiment-3 protocol

## Estimand

For each unique validation subject,

```text
H_P = BA_baseline - BA_Protected
H_N = BA_baseline - BA_MatchedNonProtected
Delta_H = H_P - H_N
```

The primary endpoint is the MEDIUM dose.  LOW, MEDIUM and HIGH are 25%, 50%
and 75% of the common train-only achievable identity-removal range.  A group
intervention is continuous in the frozen persistence coordinates:
`q'_G = (1-alpha) q_G`.

## Frozen design

* OpenBMI MI; folds 0, 1, 2; seeds 0, 1.
* V3.1 canonical full-development-TRAIN persistence spectrum and the exact
  Signed-V3.1 MI Protected block assignment are read, fingerprint-checked and
  never redefined.
* Natural Non-Protected controls are exact-rank coordinate subsets from the
  frozen non-Protected persistence-supported span.  Candidates are generated
  deterministically and selected by a predeclared train-only nearest-neighbour
  structural matching score.  Up to 50 controls are retained per Protected
  assignment; at least 20 valid controls are required.
* Matching features are: mean/dispersion of frozen rho, train coordinate
  variance/energy, train subject-ID probe BA, whitened direction norm, and
  train task-probe margin magnitude.  All are computed on development-train MI
  rows only.  No validation outcome enters matching.
* Identity calibration uses cross-session subject-ID top-1 balanced accuracy
  (session 1 -> session 2), measured on development-train subjects.  For each
  control, alpha is selected by deterministic interpolation of a fixed 21-point
  response curve to the common achievable range.  The train-only calibration
  tolerance is 1 percentage point and is not changed after freeze.
* Task probes are intervention-specific ridge probes (alpha=1e-2, same fitting
  code for baseline/P/N) fitted on all development-train MI rows and evaluated
  on validation-session-2 MI rows.  Fixed-head measurements are secondary.
* The statistical unit is unique outcome subject.  Repeated fold/seed/control
  measurements are aggregated within subject before 10,000 deterministic
  bootstrap draws.

## Gates and terminal states

* G0: LCB95 of held-out same-subject minus deterministic mismatched-subject
  cross-session persistence score is positive.
* G1: exact rank/no overlap, structural diagnostics pass, and at least 20
  controls for every Protected assignment.
* G2: validation medium-dose absolute identity drops differ by no more than
  0.01 BA (paired subject-level check).
* G3: `mean(Delta_H) >= 0.01` and its 95% bootstrap LCB is positive; the
  frozen robustness checks are reported but do not alter the estimand.

Allowed terminal states include `EXP3_UTILITY_NOT_IDENTITY_SUPPORTED`,
`PERSISTENCE_DOES_NOT_REPLICATE_ON_HELDOUT_SUBJECTS`,
`MATCHED_NONPROTECTED_CONTROL_UNAVAILABLE`, `IDENTITY_MATCH_FAILED`,
`PROTECTED_CAUSAL_EFFECT_NOT_SUPPORTED`, and
`EXP3_DEVELOPMENT_CAUSAL_EVIDENCE_ONLY`.

This is a prospectively frozen closure on a development resource already used
by earlier experiments; it is not an untouched independent replication and is
not an outer validation.
