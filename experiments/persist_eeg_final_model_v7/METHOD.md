# PERSIST-Meta: implemented V7 method

This document describes the code that was actually evaluated. It does not
describe the more ambitious architecture proposed in the original prompt.

## Episodes and base representation

For subject \(s\), legal history trials are \(H_s\) and the held-out later
session is \(Q_s\). OpenBMI uses S1 -> S2; WBCIC development uses S1/S2 -> S3.
For each outer fold, a population scaler, full-dimensional PCA, and balanced
logistic head are fit without outcome subjects. A locked raw/population logit
blend is selected using non-outcome meta episodes.

For outcome subjects, \(H_s\) labels may fit subject-specific candidate actions.
Labels in \(Q_s\) are used only to score those already-frozen actions.

## Coarse action bank

The initial bank contains 17 actions on OpenBMI and 18 on WBCIC:

- two logit-calibration actions;
- three logistic/ridge-style subject heads;
- one shrinkage-LDA head;
- one prototype-transport head;
- two full projected-gradient actions;
- eight PCA-group projected-gradient actions;
- on WBCIC only, one latest-session subject head.

Every candidate produces a future logit vector \(f_{s,m}\) from history only.
The base logits are \(f_{s,0}\). On meta-training subjects, realized utility is

\[
A^{CE}_{s,m}=CE(Q_s,f_{s,0})-CE(Q_s,f_{s,m}),
\]

with positive values indicating lower future-session cross-entropy. The audit
also records \(A^{BA}_{s,m}=BA(Q_s,f_{s,m})-BA(Q_s,f_{s,0})\).

## Generic and PERSIST context

META-GENERIC receives 14 history/component descriptors: base CE and BA, margin
and entropy statistics, prototype separation, gradient norm, session drift,
sample count, update magnitude, history-side CE/BA gain, and split-update
disagreement. Both modes also receive the component identity.

PERSIST-Meta additionally receives five quantities:

- \(P\): cosine agreement between updates estimated from two legal history
  halves;
- \(U\): leave-one-subject-out mean future CE utility for the same component
  among meta-training subjects;
- \(D\): fraction of history decisions changed by the component;
- \(G\): cosine overlap between the history logit update and the supervised
  history loss direction;
- \(R\): bidirectional CE transfer between legal history halves/sessions.

The implemented \(G\) is task-gradient overlap, not a certified protected
subspace measure. No real harmful subspace was independently certified.

## Utility controller and action rule

For each fold and mode, Ridge (alpha 1 or 10) and depth-4 ExtraTrees regressors
predict future CE utility. Predictions used for model selection are five-fold
cross-fitted by subject within the non-outcome pool. Final controllers are fit
on all non-outcome meta episodes and applied to outcome history descriptors.

Uncertainty is approximated by the meta-training standard deviation of realized
utility for each component. It is not learned jointly and is not calibrated.
For component \(m\), the score is

\[
S_{s,m}=\widehat A_{s,m}-\kappa\widehat\sigma_m-\tau.
\]

The frozen policy selects the single component with largest score. If its score
is non-positive, the action is PRESERVE. Otherwise the action is ADAPT and a
fixed scale \(c\in\{0.5,1\}\) is applied:

\[
f_s=f_{s,0}+c(f_{s,m^*}-f_{s,0}).
\]

SUPPRESS is disabled because no harmful real-EEG component was independently
certified. The evaluated controller does not compose multiple components.

Controller family, \(\kappa\in\{0,0.5,1\}\),
\(\tau\in\{0,0.0025,0.005\}\), and scale are selected by non-outcome
meta-OOF BA with harm and worst-subject tie breakers.

## Fixed anchor combinations

The selected action logits are evaluated alone and through pre-specified
combinations with the locked V6 anchor:

- residual: \(f_{anchor}+(f_{action}-f_{base})\);
- equal blend: \(0.5(f_{anchor}+f_{action})\).

These combinations do not use outcome future labels for within-run fitting.
The best PERSIST combinations were `ANCHOR_BLEND_PERSIST_META` on OpenBMI and
`ANCHOR_PLUS_PERSIST_META_RESIDUAL` on WBCIC. Neither beat the strongest generic
Conformer blend.

## Structurally distinct alternatives

The retained alternatives include history-only Euclidean alignment, a learned
filter-bank/log-variance network, compact EEG-Conformer, class-conditional
session alignment, and an eight-basis history-conditioned hypernetwork. The
hypernetwork maps history summary statistics to feature-space residual weights
and is episodically trained on non-outcome future sessions. Its generic and
PERSIST versions are capacity matched; both generalize substantially below the
anchors.

## Final status

PERSIST features improve utility prediction on average but do not establish a
performance or safety advantage. Consequently this implementation remains a
provisional research candidate and is not frozen for WBCIC outer evaluation.
