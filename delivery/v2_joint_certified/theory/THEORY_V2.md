# Joint risk-and-benefit certificate

## Assumptions

1. Episodes are exchangeable at the subject level. Training, calibration, and evaluation episodes are independent.
2. Windows within one episode may be arbitrarily dependent; each calibration subject contributes exactly one score.
3. Every action and diagnostic is measurable with respect to U. All predictors and meta-OOF scales are frozen before calibration.
4. The selector uses only U, joint bounds, fixed action availability, and fixed costs. V is opened only after decision hashing.

## Results

**Theorem 1 (simultaneous critical-index upper bound).** Applying the finite-sample split-conformal quantile to the subject maximum of normalized critical-index underestimation scores gives a marginal, actionwise simultaneous upper bound on every frozen action's critical index with probability at least 1-delta.

**Theorem 2 (simultaneous relative-benefit lower bound).** Including the normalized benefit-overestimation score for every frozen TTA action in the same subject maximum gives simultaneous lower bounds on action gain relative to No-TTA with the same marginal probability.

**Corollary (U-only post-selection).** On the joint event, any U-only selector restricted to non-sentinel risk bounds and strictly positive benefit lower bounds selects an action whose future prediction-set miscoverage is at most alpha; if it selects TTA, its future argmax-error gain relative to No-TTA is nonnegative.

This is a marginal episode-level guarantee, not conditional validity inside the certified subgroup and not a deterministic per-subject guarantee. It does not certify Macro-F1, nontrivial set existence, or the existence of a beneficial TTA. Exchangeability-breaking site shift is not covered.
