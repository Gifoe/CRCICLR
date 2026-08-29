# Subject-level D-versus-I validity reanalysis rationale

Status: **A-class required validity repair; outcome not read at protocol freeze**

## Why this repair is required

The existing Decision-Dependence-versus-Identity (D-vs-I) comparisons in Exp3,
the OpenBMI subject-invariance stress test, and the WBCIC replication used held
runs or directions as the primary uncertainty units. Those analyses establish
algorithmic reproducibility across frozen runs. Runs, seeds, directions,
configurations, and backbone rows are not biological subjects, so their
confidence intervals cannot be presented as subject-population intervals.

This is a validity problem, not an invitation to search for a stronger result.
The repair has one question only:

> Across biological outcome subjects, does a source-side Decision Dependence
> model have lower doubly cross-fitted consequence-prediction error than the
> matched Subject Identity model?

## A/B/C classification

- **Class A — required:** reconstruct subject-level intervention consequences
  for the retained OpenBMI stress-test and WBCIC development runs, refit the
  unchanged diagnostic models with simultaneous subject/run exclusion inside
  the subject's clean historical outcome fold, and make the biological subject
  the sole inferential unit.
- **Class B — useful but not required:** a new independently sealed cohort.
  This repair does not authorize one.
- **Class C — unnecessary:** retraining encoders, changing directions, tuning
  ridge penalties, adding predictors, selecting configurations, or opening any
  sealed/outer resource.

## Feasibility established without reading new outcomes

Read-only schema inspection established that the historical development
runtimes retain, for every required run, the frozen embedding archive, frozen
task-head checkpoint, source-defined direction record, and fold metadata. The
original code computes per-subject effects before averaging but did not retain
them in its direction table. The retained artifacts are therefore sufficient
to reconstruct those effects without retraining or accessing a new outcome
resource.

An initially considered leave-one-outcome-fold-out repair was rejected before
execution. Subjects held out as outcomes in one historical fold re-enter source
or model-fit roles in other folds, so pooling other folds would leak the test
subject's data into predictor inputs. The frozen analysis instead stays inside
the one historical fold whose encoder and source directions excluded the
entire outcome-subject set. Within that fold it simultaneously leaves out (i)
one biological outcome subject and (ii) one complete algorithmic run. Predictor
training uses only the other outcome subjects and the other runs. No other fold
is allowed to enter that subject's fit.

The exact historical scopes are:

- OpenBMI: 40 already-observed development subjects, five frozen outcome folds,
  two backbones, three seeds, the complete ten-configuration stress grid, and
  eight source-defined directions per configuration/run.
- WBCIC: 41 already-observed development subjects, five frozen outcome folds,
  three seeds, the frozen ERM direction audit, and eight source-defined
  directions per run.

The OpenBMI primary averages ERM, DANN, CORAL, and MMD method families with
equal total family weight. WBCIC has only the frozen ERM bank. These are two
predeclared dataset-specific estimands, not an exactly matched intervention-bank
replication. The OpenBMI ERM-only analysis is therefore mandatory as a
direct-scope sensitivity, and the equal-configuration historical grid is a
second mandatory weighting sensitivity. Neither sensitivity may replace the
primary after outcomes are known.

The numerical repair is applied to both sides of the comparison. For OpenBMI,
the historical direction table mixed stored lower-precision intact logits with
float64 logits recomputed after erasure. The locked implementation must first
reproduce the historical consequence and `D_finite` values as integrity checks,
then recompute intact and erased outcome logits and intact and erased source
logits through the same frozen head in float64. Only the matched-float64
subject consequence and matched-float64 `D_finite` enter the locked analysis.

OpenBMI internal-14 remains `INVALID_FOR_CONFIRMATION`; WBCIC outer-10 remains
`OUTER_UNTOUCHED`. Neither is an input. Exp3 cannot be repaired because its
per-subject runtime was not retained; its D-vs-I interval will be labeled
run-level/algorithmic only.

## Non-negotiable interpretation rule

The outcome sign is binding. If the subject-level estimate weakens, reverses,
or becomes uncertain, the manuscript claim must shrink accordingly. The old
run-level interval remains an algorithmic reproducibility result and cannot be
used to override the subject-level estimate.

This is a retrospective, conditional validity audit over the already-frozen
development intervention grid. It is not a new independent confirmation, and
it does not estimate performance for arbitrary future interventions.

The exact estimand, cross-validation, aggregation, resampling, decision gates,
inputs, and stop rules are frozen in
`protocol/SUBJECT_LEVEL_D_VS_I_REANALYSIS_LOCK.json`. The implementation and
input manifest must be committed in the same pre-outcome lock commit. The run
also requires the synthetic test file to be tracked and clean in that commit,
refuses to overwrite an existing output, and publishes the six required result
artifacts only by renaming a completed staging directory after validation.

Before any input manifest was created or any subject-level outcome was loaded,
an independent code audit identified two enforcement gaps and they were closed:
(i) point estimates and percentile intervals with opposite directions now enter
an explicit `POINT_CI_DIRECTION_CONFLICT` gate instead of being silently mapped
to non-support or reversal; and (ii) the protocol, manifest, and one-shot output
directory are bound to their canonical repository paths, so changing a CLI path
cannot create a second scientific run. The same pre-outcome hardening also adds
strict finite-number serialization, an exact WBCIC ten-configuration
source-freeze check, atomic manifest publication, and same-file-handle SHA-256
verification immediately before any pickle-enabled historical NPZ load.
