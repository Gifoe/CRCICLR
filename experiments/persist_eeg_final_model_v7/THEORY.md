# Compact theoretical account

The statements below are deliberately narrow. They formalize the decision
logic; they are not claims that the V7 estimator is statistically consistent or
that P/U/D/G/R causally identify future utility.

## 1. Predictability does not imply optimal invariance

Let a representation be \(Z=(Z_T,Z_S)\), where \(Z_S\) predicts subject
identity and \(Y\) is the task label. Subject predictability means only that
\(I(Z_S;S)>0\). It does not imply \(I(Z_S;Y\mid Z_T)=0\).

**Proposition 1.** There are distributions for which subject identity is
predictable from \(Z_S\), yet every representation invariant to \(S\) has
higher Bayes task risk than a non-invariant representation.

**Construction.** Let subjects have different but stable task calibrations,
and let \(Z_S\) identify which calibration applies. Given \((Z_T,Z_S)\), the
classifier can apply the correct calibration. Removing all subject information
forces a mixture calibration and increases risk whenever the calibrations
differ on a set of positive probability. Thus identity predictability alone is
not evidence that a direction is nuisance.

This proposition does not say subject information is always useful. It says an
erasure decision requires utility evidence beyond an identity probe.

## 2. Conditional selection under heterogeneous utility

Let \(A_m\) be the future loss reduction from applying component \(m\), with
\(m=0\) denoting PRESERVE and \(A_0=0\). Let \(X\) be legal history context.
Assume the controller may choose one component and that no additional action
cost is present.

**Proposition 2.** The Bayes conditional policy

\[
\pi^*(X)=\arg\max_{m\in\{0,\ldots,M\}} E[A_m\mid X]
\]

has expected utility at least that of every global fixed action. The inequality
is strict when different actions uniquely maximize conditional expected utility
on positive-probability regions of \(X\).

**Reason.** Pointwise maximization gives

\[
\max_m E[A_m\mid X]\ge E[A_k\mid X]
\]

for every fixed \(k\); taking expectations preserves the inequality. Strict
heterogeneity yields strict improvement.

The proposition gives a reason to learn a conditional policy, not a guarantee
that finite-sample estimates will beat a global action. V7's negative BA result
is compatible with estimation error, weak action headroom, and distribution
shift between meta and outcome subjects.

## 3. Risk-aware adaptation with asymmetric harm

Suppose adaptation yields uncertain utility \(A_m\), changing a component has
cost \(c_m\), and harmful updates receive additional penalty \(\lambda>0\).
One rational policy maximizes

\[
E[A_m\mid X]-c_m-\lambda\,\mathcal R(A_m\mid X),
\]

where \(\mathcal R\) is a chosen downside-risk functional. Under a
mean-standard-deviation surrogate, this becomes

\[
S_m=\mu_m-\kappa\sigma_m-c_m.
\]

**Proposition 3.** Relative to PRESERVE with zero utility, component \(m\)
should be applied only when \(S_m>0\), provided the mean-standard-deviation
objective correctly represents the decision maker's harm preference.

This is an optimization identity under an assumed objective, not a theorem that
standard deviation is the correct EEG risk measure. In V7, \(\sigma_m\) is an
empirical component-level standard deviation rather than calibrated
subject-specific uncertainty, so the risk score should be treated as a
heuristic lower-confidence surrogate.

## 4. Role of PERSIST descriptors

P, U, D, G, and R are context variables for estimating
\(E[A_m\mid X]\). They do not determine the action by definition. In
particular:

- high persistence does not imply preserve or adapt;
- decision dependence does not imply safe removability;
- signed historical transfer does not guarantee future transfer;
- task-gradient overlap is not causal certification of a protected subspace.

Their empirical value must be judged by out-of-subject utility prediction and
downstream future-session performance against a capacity-matched context. V7
finds a small predictive increment but no resolved downstream gain. Therefore
the theory motivates the test; it does not rescue the failed empirical target.
