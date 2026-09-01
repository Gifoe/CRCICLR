# PERSIST-CDE SEED-0 PILOT

**Competence-Preserving Counterfactual Decision Ensemble (PERSIST-CDE)** is a
single predeclared pilot.  It asks one narrow question: can two lightweight
counterfactual decision branches improve future-session balanced accuracy
when the already competent canonical EEGNet remains an immutable anchor?

Existing project evidence does not justify the blanket claim that
subject-identifiable information is nuisance information.  Persistent
information can be task useful; identity suppression and SCST/PDA/U-PDA or
PERSIST-RE routes did not produce stable prospective utility; and aggressive
routing did not generalize reliably.  However, competent predictors can make
complementary decision errors, while probability averaging can be useful.
Therefore this pilot preserves the competent predictor and treats invariance
and geometry only as auxiliary counterfactual decision hypotheses:

> Preserve the competent predictor as an immutable anchor.  Treat
> invariance/geometric interventions only as auxiliary counterfactual decision
> hypotheses, and use them only through conservative residual fusion.

The canonical EEGNet backbone and baseline head are frozen during adapter
training.  INV uses a gradient-reversal subject discriminator and a fixed
KL-to-baseline penalty.  GEO uses only class-conditional subject CORAL, with
no covariance term when a subject/class cell has fewer than two samples.  Both
branches are `LayerNorm(64) -> Linear(64,16) -> GELU -> Linear(16,64)` with a
zero-initialized last layer and residual scale 0.25; their heads start as
copies of the baseline head.  Thus step zero is the baseline and the method
can退化 to `OURS = BASELINE`.

The only trainable data are legal model-fit subjects during development and
model-fit plus discovery subjects during final refit.  Fusion is a fold-level
constant selected on discovery subjects using the predeclared robust loss and
one-SE/minimal-residual rule in `PROTOCOL_LOCK.json`.  B1/B2 are diagnostic
controls only and cannot replace B4 after outcome scoring.

This is seed 0 only.  It does not search architectures, coefficients, epochs,
or seeds, and it does not access WBCIC's sealed outer ten or any OpenBMI
sealed/internal holdout.  A positive seed-0 result is a pilot signal, not a
multi-seed paper claim.
