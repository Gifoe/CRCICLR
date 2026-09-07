# HSC-TTA critical-index certificate

## Formal target

For subject `s`, action `a`, and lambda `lambda_j`, the formal risk is the empirical prediction-set miscoverage on the pre-fixed future deployment episode:

`R_s^V(a, lambda_j) = mean_{i in V_s} 1[y_i not in C_{a,lambda_j}(x_i)]`.

The certificate controls this finite, predefined episode-level empirical quantity. It does not control an infinite-time latent subject risk or a stationary population expectation. The empirical-Bernstein block bound remains supplementary diagnostic code only; it is not a training target, conformal score, feasibility condition, or paper-level main certificate.

## Episode and information boundary

Each deployment episode is `E_s=(U_s,V_s)`. Features, test-time adaptation, critical-index prediction, certification, and action selection use `U_s` only. Outcomes from `V_s` are available only for historical meta-risk supervision, calibration scores, and final offline evaluation after the final-test decision table is frozen.

HMC and CAP use a 90-minute context followed by the first 240 valid scored 30-second epochs (`V_s-main`). EEGMMIDB uses runs 4 and 6 for context and runs 8, 10, 12, and 14 for future evaluation. The remaining sleep night is retained as `V_s-full` for supplementary robustness only.

## Critical index

The grid contains 20 nontrivial lambdas followed by `lambda_L=1.0`. The sentinel produces the full class set and therefore has empirical risk zero. For fixed alpha,

`J_s(a,alpha) = min {j : R_s^V(a,lambda_j) <= alpha}`.

The sentinel guarantees existence. `J_s=L` means no nontrivial lambda meets the target.

An alpha-specific low-capacity predictor uses only `U_s` features and action diagnostics:

`J_hat_s(a,alpha) = g_alpha(h_s, a)`.

Predictors are fitted on meta-risk-training subjects only. Hyperparameters use subject-grouped cross-validation inside that role. Calibration and final-test subjects are excluded. CAP inherits the corresponding HMC predictor.

## Actionwise simultaneous conformal correction

For calibration subject `s` and fixed alpha,

`E_s(alpha) = max_a [J_s(a,alpha) - J_hat_s(a,alpha)]`.

The maximum is over the three actions only: `no_tta`, `t3a`, and `entropy_adapter`. It is never taken over lambda indices. With `m` calibration subjects,

`k = ceil((m+1)(1-delta))`.

If `k<=m`, `raw_q_alpha` is the `k`th sorted subject score and `q_alpha=max(0,raw_q_alpha)`. If `k>m`, the implementation records a conservative full-set-index fallback rather than silently reporting success.

For a new subject,

`J_bar_s(a,alpha) = clip(ceil(J_hat_s(a,alpha)+q_alpha),0,L)`.

An action is a nontrivial certified candidate only when `J_bar<L`. The selected lambda is `lambda[J_bar]`. `J_bar=L` is an uncertified full-set fallback and never contributes to nontrivial CSR.

## Guarantee and assumptions

The guarantee assumes exchangeability of subject episodes within the relevant calibration/test population. The fitted predictor and all preprocessing choices must be independent of calibration and test subjects. Under these assumptions, with subject-level probability at least `1-delta`, simultaneously for all candidate actions,

`R_*^V(a, lambda_{J_bar_*(a,alpha)}) <= alpha`.

Because the event is simultaneous across actions, any action selection rule that depends only on `U_s` information and the certified candidate table preserves the same guarantee. The result is separately calibrated for alpha 0.10 and alpha 0.20; no joint-alpha guarantee is claimed. Any supplementary joint-alpha analysis requires an explicit multiplicity correction such as Bonferroni.
