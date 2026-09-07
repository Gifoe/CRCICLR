# PERSIST-EEG Experiment 3 V1.1 protocol

## Scientific question and estimand

The experiment tests whether deleting the frozen Signed-V3.1 Protected
persistent structure causes more task degradation than deleting a structurally
matched Non-Protected persistent structure at the same held-out
cross-session subject-identity reduction.  The causal unit is one frozen
Protected block per run; V1's union-level unit is not reused.

For validation subject `s`, run `r`, Protected block `b`:

```text
H_P(s,r,b) = BA_baseline - BA_Protected
H_N(s,r,b) = BA_baseline - BA_MatchedNonProtectedEnsemble
Delta_H(s,r,b) = H_P - H_N
```

Within a run, block effects are averaged; repeated controls are averaged
before block aggregation; repeated appearances of a subject are averaged
before the 10,000-draw subject bootstrap.  The primary endpoint is MEDIUM
dose, with LOW/MEDIUM/HIGH equal to 25/50/75% of each Protected block's
train-only `Dmax_P` identity reduction.  Interventions use
`q'_G=(1-alpha)q_G` in the frozen canonical coordinates.

## Frozen upstream and data boundary

OpenBMI MI, folds 0–2 and seeds 0–1 use the Signed-V3.1 EEGNet checkpoint,
canonical spectrum, whitening, directions, rho, blocks, persistence support and
MI Protected assignment without redefinition.  Development-train subjects
alone determine candidates, persistence certification, matching, identity
metric, alpha/dose targets, probes and gates.  Development-validation subjects
are read only after freeze for held-out persistence, identity manipulation and
task outcomes.  Outer EEG, labels, features, subject IDs and membership are
never loaded or enumerated; all artifacts must set both outer flags false.

## Block-wise controls

For every frozen Protected block `P_{r,b}`:

1. Candidate coordinates are exact-rank subsets of the non-Protected,
   persistence-supported canonical span, with zero overlap with every frozen
   Protected block in that run.
2. All natural combinations are enumerated when feasible; otherwise the
   SHA256-seeded candidate sample and every sampled coordinate set are persisted.
3. A candidate first passes train-only persistence certification
   `R_persist_train > 0`, where the statistic is same-subject minus
   deterministic shifted-subject cross-session cosine.
4. Certified candidates are standardized on the candidate pool and ranked by
   the frozen structural distance over rho mean/dispersion, coordinate
   variance/energy, train subject-ID BA, dewhitened direction norm and train
   task-margin magnitude.  The first 50 are retained; at least 20 are required.

Duplicate rotations of a full-rank ambient span are not controls.  A
rotation/Grassmann fallback would only be legal before freeze when ambient
dimension is strictly larger than rank, every generated basis is persisted and
the same train persistence certification is applied.  No fallback may admit
unsupported coordinates.

## Identity calibration and task outcome

The primary train identity metric is the mean of S1→S2 and S2→S1 subject-ID
balanced accuracy; human-readable directional BA values are retained.  For
each Protected block, `Dmax_P` is its maximum positive train identity drop.
Each N control independently solves the predeclared P-anchored LOW/MEDIUM/HIGH
targets by deterministic interpolation.  Controls unable to reach a target are
ineligible at that dose; MEDIUM must retain at least 20 controls per eligible
block.

After freeze, all baseline, Protected and N probes are intervention-specific
ridge probes (`alpha=1e-2`) fitted on identical development-train MI rows and
evaluated on validation-session-2 MI rows.  Same-rank random controls are
secondary sanity diagnostics; Neutral-only controls, if reported, never affect
the primary gate.

## Gates and terminal states

* G0: held-out persistence `LCB95(R_persist)>0`.
* G1: at least five of six runs eligible, at least 80% of frozen Protected
  blocks eligible, every eligible block has exact rank/no overlap, train
  persistence evidence and at least 20 MEDIUM controls.
* G2: validation MEDIUM requires both arms to produce measurable identity
  reduction (`mean(Delta_ID_P) > 0` and `mean(Delta_ID_N) > 0`) and requires
  `|mean(Delta_ID_P)-mean(Delta_ID_N)| <= 0.01 BA`.  The strict positive-drop
  clause is necessary: an arm with zero reduction cannot establish a matched
  identity-removal comparison.
* G3: mean MEDIUM `Delta_H >= 0.01 BA`, 95% bootstrap lower bound > 0, at
  least 5/6 eligible run means positive, and at least 60% of unique subjects
  nonnegative.  These rules are frozen before validation outcomes.

Allowed scientific terminal states are:

```text
EXP3_UTILITY_NOT_IDENTITY_SUPPORTED_DEVELOPMENT
PROTECTED_CAUSAL_EFFECT_NOT_SUPPORTED
IDENTITY_MATCH_FAILED
EXP3_INSUFFICIENT_BLOCKWISE_CONTROL_COVERAGE
PERSISTENCE_REPLICATION_FAILED
MEASUREMENT_INVALID
PROTOCOL_VIOLATION
```

The final report must answer all requested block coverage, identity-dose,
causal effect, dose-response, random/Neutral-only, leakage and claim questions.
It must not call this an independent or untouched replication.  `YES` for the
utility-not-identity claim and `READY_FOR_EXPERIMENT_4=YES` are allowed only if
all primary gates pass.
