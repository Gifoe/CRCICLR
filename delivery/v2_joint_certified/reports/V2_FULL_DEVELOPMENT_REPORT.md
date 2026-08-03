# HSC-TTA v2 full development report

## Verdict

The implementation and nested development protocol are complete, but the empirical main-method claim fails. The simultaneous certificate is conservative and valid on development episodes, yet it selects no TTA in either task and captures none of the available Safe-Oracle gain. This is a **NO-GO for an ICLR main-method submission in its current form**.

## Proposed-policy main table

| dataset   |   alpha | policy           |   argmax_error |   average_set_size |    csr |   full_set_fallback |   joint_validity |   macro_f1 |   marginal_violation |   safe_oracle_gain_captured |   selected_tta_ppv |   selected_vs_no_tta_gain |   tta_selection_rate |
|:----------|--------:|:-----------------|---------------:|-------------------:|-------:|--------------------:|-----------------:|-----------:|---------------------:|----------------------------:|-------------------:|--------------------------:|---------------------:|
| eegmmidb  |  0.1000 | joint_hsc_tta_v2 |         0.5995 |             3.9678 | 0.2711 |              0.7289 |           1.0000 |     0.3925 |               0.0000 |                      0.0000 |                nan |                    0.0000 |               0.0000 |
| eegmmidb  |  0.2000 | joint_hsc_tta_v2 |         0.5995 |             3.8777 | 0.4978 |              0.5022 |           1.0000 |     0.3925 |               0.0000 |                      0.0000 |                nan |                    0.0000 |               0.0000 |
| hmc       |  0.1000 | joint_hsc_tta_v2 |         0.3884 |             4.8600 | 0.0764 |              0.9236 |           0.9891 |     0.4757 |               0.0109 |                      0.0000 |                nan |                    0.0000 |               0.0000 |
| hmc       |  0.2000 | joint_hsc_tta_v2 |         0.3884 |             4.7178 | 0.1236 |              0.8764 |           0.9964 |     0.4757 |               0.0036 |                      0.0000 |                nan |                    0.0000 |               0.0000 |

`selected_tta_ppv` is undefined when no TTA is selected. Full-set fallback is reported as fallback, never counted as a nontrivial certificate.

## Independent policy baselines

| dataset   |   alpha | policy                    |   argmax_error |   average_set_size |    csr |   full_set_fallback |   macro_f1 |   marginal_violation |   safe_beneficial_selection_precision |   selected_vs_no_tta_gain |   tta_selection_rate |
|:----------|--------:|:--------------------------|---------------:|-------------------:|-------:|--------------------:|-----------:|---------------------:|--------------------------------------:|--------------------------:|---------------------:|
| eegmmidb  |  0.1000 | agreement_gate_policy_crc |         0.5996 |             3.8397 | 0.6400 |              0.3600 |     0.3918 |               0.0578 |                                0.1461 |                   -0.0001 |               0.9778 |
| eegmmidb  |  0.1000 | best_fixed_policy_crc     |         0.6015 |             3.7802 | 0.7600 |              0.2400 |     0.3889 |               0.0444 |                                0.2654 |                   -0.0020 |               0.7200 |
| eegmmidb  |  0.1000 | entropy_gate_policy_crc   |         0.6015 |             3.7802 | 0.7600 |              0.2400 |     0.3889 |               0.0444 |                                0.2654 |                   -0.0020 |               0.7200 |
| eegmmidb  |  0.1000 | no_tta_global_crc         |         0.5995 |             3.7624 | 1.0000 |              0.0000 |     0.3925 |               0.0311 |                              nan      |                    0.0000 |               0.0000 |
| eegmmidb  |  0.2000 | agreement_gate_policy_crc |         0.5996 |             3.6133 | 1.0000 |              0.0000 |     0.3918 |               0.0400 |                                0.1506 |                   -0.0001 |               0.9778 |
| eegmmidb  |  0.2000 | best_fixed_policy_crc     |         0.6015 |             3.4427 | 1.0000 |              0.0000 |     0.3889 |               0.0356 |                                0.2654 |                   -0.0020 |               0.7200 |
| eegmmidb  |  0.2000 | entropy_gate_policy_crc   |         0.6015 |             3.4427 | 1.0000 |              0.0000 |     0.3889 |               0.0356 |                                0.2654 |                   -0.0020 |               0.7200 |
| eegmmidb  |  0.2000 | no_tta_global_crc         |         0.5995 |             3.4229 | 1.0000 |              0.0000 |     0.3925 |               0.0356 |                              nan      |                    0.0000 |               0.0000 |
| hmc       |  0.1000 | agreement_gate_policy_crc |         0.3887 |             3.7233 | 0.6800 |              0.3200 |     0.4751 |               0.0473 |                                0.1594 |                   -0.0003 |               0.9818 |
| hmc       |  0.1000 | best_fixed_policy_crc     |         0.3935 |             3.9476 | 0.5600 |              0.4400 |     0.4733 |               0.0509 |                                0.2500 |                   -0.0051 |               0.4800 |
| hmc       |  0.1000 | entropy_gate_policy_crc   |         0.3932 |             3.9499 | 0.5600 |              0.4400 |     0.4734 |               0.0509 |                                0.2460 |                   -0.0048 |               0.4036 |
| hmc       |  0.1000 | no_tta_global_crc         |         0.3884 |             3.7255 | 0.6800 |              0.3200 |     0.4757 |               0.0473 |                              nan      |                    0.0000 |               0.0000 |
| hmc       |  0.2000 | agreement_gate_policy_crc |         0.3887 |             2.9395 | 0.9600 |              0.0400 |     0.4751 |               0.0582 |                                0.1594 |                   -0.0003 |               0.9818 |
| hmc       |  0.2000 | best_fixed_policy_crc     |         0.3935 |             3.1430 | 0.8800 |              0.1200 |     0.4733 |               0.0582 |                                0.2348 |                   -0.0051 |               0.4800 |
| hmc       |  0.2000 | entropy_gate_policy_crc   |         0.3932 |             3.0662 | 0.9200 |              0.0800 |     0.4734 |               0.0582 |                                0.2301 |                   -0.0048 |               0.4036 |
| hmc       |  0.2000 | no_tta_global_crc         |         0.3884 |             2.8658 | 1.0000 |              0.0000 |     0.4757 |               0.0545 |                              nan      |                    0.0000 |               0.0000 |

The agreement policy is a custom U-only agreement heuristic with policy CRC; it is not represented as official TTALine. Tent/EATA are architecture-incompatible because the selected CBraMod heads use LayerNorm rather than adaptable BatchNorm; no fake official implementation is reported.

## Predictor audit

### Benefit

| dataset   | model                  |   gain_mae |   sign_balanced_accuracy |   spearman |
|:----------|:-----------------------|-----------:|-------------------------:|-----------:|
| eegmmidb  | agreement_only         |     0.0145 |                   0.5208 |     0.0750 |
| eegmmidb  | constant_zero          |     0.0135 |                   0.5000 |   nan      |
| eegmmidb  | elastic_net            |     0.0183 |                   0.4935 |    -0.0669 |
| eegmmidb  | entropy_only           |     0.0145 |                   0.5518 |     0.1228 |
| eegmmidb  | global_mean            |     0.0145 |                   0.5000 |   nan      |
| eegmmidb  | hist_gradient_boosting |     0.0155 |                   0.4509 |    -0.1977 |
| eegmmidb  | pairwise_sign_gain     |     0.0268 |                   0.4858 |    -0.0144 |
| hmc       | agreement_only         |     0.0240 |                   0.5074 |     0.1581 |
| hmc       | constant_zero          |     0.0231 |                   0.5000 |   nan      |
| hmc       | elastic_net            |     0.0278 |                   0.4896 |     0.1131 |
| hmc       | entropy_only           |     0.0250 |                   0.5471 |     0.1099 |
| hmc       | global_mean            |     0.0249 |                   0.5000 |   nan      |
| hmc       | hist_gradient_boosting |     0.0282 |                   0.4904 |     0.1421 |
| hmc       | pairwise_sign_gain     |     0.0386 |                   0.4968 |     0.0586 |

### Risk

| dataset   |   alpha | model                   |    mae |   underestimation_rate |
|:----------|--------:|:------------------------|-------:|-----------------------:|
| eegmmidb  |  0.1000 | action_mean             | 1.0267 |                 0.5483 |
| eegmmidb  |  0.1000 | constant_critical_index | 1.2100 |                 0.3567 |
| eegmmidb  |  0.1000 | elastic_net             | 1.6208 |                 0.5094 |
| eegmmidb  |  0.1000 | hist_gradient_boosting  | 1.1166 |                 0.5172 |
| eegmmidb  |  0.2000 | action_mean             | 1.3475 |                 0.5250 |
| eegmmidb  |  0.2000 | constant_critical_index | 1.4633 |                 0.3861 |
| eegmmidb  |  0.2000 | elastic_net             | 2.1196 |                 0.5011 |
| eegmmidb  |  0.2000 | hist_gradient_boosting  | 1.4468 |                 0.4967 |
| hmc       |  0.1000 | action_mean             | 2.4470 |                 0.5498 |
| hmc       |  0.1000 | constant_critical_index | 2.7062 |                 0.4471 |
| hmc       |  0.1000 | elastic_net             | 3.3329 |                 0.4987 |
| hmc       |  0.1000 | hist_gradient_boosting  | 2.9371 |                 0.5502 |
| hmc       |  0.2000 | action_mean             | 3.7736 |                 0.5129 |
| hmc       |  0.2000 | constant_critical_index | 3.8387 |                 0.4573 |
| hmc       |  0.2000 | elastic_net             | 5.3661 |                 0.5098 |
| hmc       |  0.2000 | hist_gradient_boosting  | 4.6141 |                 0.5244 |

## Direct answers

1. **Are the repaired source models qualified?** Yes for development use: HMC temporal attention and EEGMMIDB official all-patch downstream heads beat weak/majority behavior across five seeds and EEGMMIDB predicts all four classes. This qualification does not imply the selector succeeds.
2. **How often are actions truly better than No-TTA?** The action audit records substantial raw beneficial cases (T3A roughly 35–39% depending on task/alpha), while robust-residual benefit is much rarer (roughly 7–13%). Exact per-action values are in `ACTION_WIN_RATE.csv`.
3. **How large is Safe-Oracle headroom?** Mean development Safe-Oracle gain is approximately 0.0173 for HMC and 0.0120 for EEGMMIDB (dataset aggregation over stored action-audit rows). It exists but is small.
4. **Does the benefit predictor beat simple surrogates?** Not reliably. ElasticNet sometimes improves sign discrimination, but constant-zero often has equal or lower gain MAE. The positive-gain lower bound therefore remains non-positive for every selected candidate.
5. **Is the risk predictor accurate enough?** It has usable ranking/MAE for conservative bounds, but calibration inflation is large. Risk CSR is nonzero, particularly at alpha=0.20, while many subjects still require the sentinel full set.
6. **Does the joint certificate reach nominal validity?** Yes at the predeclared 0.90 level in nested development: EEGMMIDB is 1.000 and HMC is 0.989/0.996 for alpha 0.10/0.20; simulation mean simultaneous validity is 0.977. This is marginal episode-level evidence, not conditional or per-subject certainty.
7. **Is joint calibration tighter than separate calibration?** No consistent advantage is established. The separate-calibration ablation is often less conservative; that does not give it the proposed simultaneous post-selection theorem.
8. **Does the proposed policy beat No-TTA+CRC, Best-Fixed+CRC, Entropy Gate+CRC, and agreement+CRC?** No. Its argmax predictions equal No-TTA because no TTA is selected; several heuristic/fixed policies trade validity or utility differently, but the proposed method has no positive utility advantage.
9. **Does it capture Safe-Oracle gain?** No: Safe-Oracle gain captured is 0.000 in all four dataset/alpha blocks.
10. **What fraction selects TTA?** 0.000 in nested development for both tasks and both alphas.
11. **What fraction of selected TTA is truly beneficial?** Undefined because the selected-TTA denominator is zero; it must not be reported as 0% or 100% precision.
12. **Does safety rely on full-set fallback?** Materially yes. EEGMMIDB fallback is high at alpha=0.10 and remains substantial at alpha=0.20; HMC also uses fallback. Exact rates are in the main table.
13. **Is calibration size sufficient?** It is sufficient to obtain conservative marginal validity, not sufficient for useful positive-benefit certification. The m=12/14 folds cannot overcome weak benefit prediction; larger m only addresses quantile granularity.
14. **Which dataset still fails and why?** Both fail the adaptive-utility objective. EEGMMIDB additionally suffers higher error and larger fallback; HMC has real action headroom but the benefit certificate does not identify it.
15. **Are the results sufficient to proceed to a new confirmatory dataset?** No for a confirmatory claim. A new dataset should only be acquired after improving U-only benefit predictability or action reliability on untouched development data and re-freezing the method.

## Theoretical scope

The theorem controls marginal episode-level risk and non-harm after an arbitrary U-only selector using a simultaneous subject score. It does not guarantee conditional validity for the certified subgroup, deterministic safety for every subject, Macro-F1, a non-full prediction set, or existence of a beneficial TTA action.

## Reproducibility and taint

All method selection used source-fit and v2 nested-development subjects. Old final outcomes were accessed only by the one-time v1 diagnosis before development and, after `V2_METHOD_FREEZE.json`, by separately labeled exploratory replication. Those replications are not confirmatory and cannot be used to revise v2.

A post-freeze evaluator audit corrected risk/set metrics to use each frozen certified index instead of the oracle true critical index. No predictor, action, bound, q, selector, or decision changed; hashes before/after and the unchanged decision hash are recorded in `V2_EVALUATION_CORRECTION.json`.
