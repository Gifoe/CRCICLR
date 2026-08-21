# Frozen scientific protocol (candidate V1)

## Question and scope

The experiment tests the chain `identity reduction -> protected-persistence
loss -> task harm -> selective rescue`. It uses OpenBMI/Lee2019 MI only. It
does not use the V6--V8 generic adaptation, Conformer blend, action bank,
router, or meta-selector machinery.

All outputs are `DEVELOPMENT / EXPLORATORY`. The first three frozen development
folds and seeds 0 and 1 are evaluated. The statistical unit is subject.

## Data separation

For each fold, the existing 34 `train_subjects` are deterministically divided
into 28 model-fit and 6 calibration subjects by SHA256 ordering. The existing
9 `validation_subjects` are the outcome subjects. Only these two JSON fields
are read. Model selection, normalization, protected assignment, rescue fitting,
and rescue scale selection use model-fit/calibration data; outcome Session 2
labels are used only for final scoring. Subject probes fit on outcome Session 1
identity labels and score Session 2 identity labels.

## Frozen method roster

- `A0_TASK_ONLY_EEGNET`: EEGNet task cross-entropy only.
- `A1_SUBJECT_GRL_EEGNET`: identical EEGNet and subject discriminator with a
  fixed GRL ladder `lambda={0.05,0.1,0.3,1.0}`. Lambda 0.3 is the primary A1
  comparison; the full ladder is retained for Figure A.
- `B0_EEG_DG_TASK_ONLY` / `B1_EEG_DG_FULL`: identical multi-scale routed
  architecture. Full adds marginal RBF-MMD, class-conditional centroid
  alignment, and source-domain classification at weights 0.1/0.1/0.1.
- `C0_SCLDGN_TASK_ONLY` / `C1_SCLDGN_FULL`: identical fixed nine-band FIR plus
  multi-kernel spatial-temporal architecture. Full adds pairwise CORAL and
  same-class feature-mix supervised contrastive loss at weights 1.0/0.1.

The B/C implementations are clean-room method-level reproductions because the
official repositories have no license file. Deviations are not hidden and are
listed in `LICENSE_AUDIT.md` and `METHOD_FIDELITY.md`.

## Fidelity before science

Every roster member runs fold 0 / seed 0 smoke training. Task-only calibration
BA must exceed 0.60, shapes and finite losses must pass, task-only/full models
must have equal parameter counts within family, and no outcome loader may be
constructed during training. Only implementation/fidelity repair is permitted
before `PROTOCOL_FROZEN.json` is emitted. After freezing, only demonstrable bug
repair is allowed; gates and scientific definitions cannot change.

## Invariance and protected audits

Subject information is a low-capacity multinomial logistic probe fit on
outcome Session 1 representations and scored on Session 2. The primary effect
is `DeltaID = ID_invariant - ID_task_only`.

For each family/run, the task-only representation defines a Signed-V3.1-style
whitened cross-session persistence spectrum on model-fit subjects. Eigengap
blocks have maximum rank four. Persistence support uses 200 deterministic
subject/session permutations. Signed utility uses five deterministic
subject-disjoint inner splits, 32 trials per subject/session/class, 100
same-rank random erasures, and 10,000 subject bootstrap draws. A block is
Protected only when persistence is supported and both the absolute and
random-calibrated erasure-harm CI lower bounds are positive.

Protected retention is the cross-subject/cross-session variance-weighted R2 of
a ridge map from a representation to frozen task-only protected coordinates:
fit on model-fit Session 1, evaluate on outcome Session 2. CCA/cosine geometry
and matched non-Protected retention are secondary diagnostics.

The pre-registered gates are:

- I1: mean `DeltaID < 0`.
- I2: mean `DeltaPRS < 0`.
- I3: mean `DeltaBA_INV < 0`.

Only a family satisfying I1+I2+I3 is `ELIGIBLE_PROTECTED_LOSS` and enters the
primary rescue analysis.

## Rescue and controls

Task-only and invariant encoders remain frozen. Each same-rank rescue learns a
ridge residual map `B` from teacher coordinates to `h_task-h_invariant`, then
fits the same balanced ridge task head. Ridge `{0.001,0.1,1.0}` and residual
scale `{0.25,0.5,1.0}` are selected on calibration subjects only.

- invariant only;
- 20 deterministic random orthogonal task-teacher subspaces;
- high-persistence directions selected without signed utility;
- task-teacher PCA directions;
- PERSIST Protected directions;
- full task-teacher representation as a diagnostic upper bound.

All rank-matched residuals use identical access, rank, trainable residual
parameter count, head, optimizer-equivalent closed-form criterion, and stopping
rule. Random results are reported as a distribution, never cherry-picked.

Minimum support is paired subject mean `PERSIST-invariant > 0`. Certified
support requires both 95% lower bounds `PERSIST-invariant > 0` and
`PERSIST-generic > 0`. Recovery ratio >=0.50 is a descriptor, not a selector.

## Statistics and claims

All primary confidence intervals use 10,000 deterministic hierarchical draws
over folds, seeds, and subjects. Primary family p-values are subject-level
paired sign-flip tests with Holm correction across A/B/C. Trial-level bootstrap
is forbidden. All lambdas, runs, random draws, failures, and exclusions remain
in the ledgers.

Cross-family support requires at least two certified eligible families and a
positive pooled hierarchical lower bound versus generic persistence. No
scientific gate may be changed after freezing.

`OUTER_TEST_USED=false` is mandatory. The outer split field is not read; outer
signals, labels, features, and metrics are not constructed.

