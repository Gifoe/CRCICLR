# PERSIST-EEG V4 scientific report

## Executive decision

Terminal state: `GENERIC_DYNAMIC_ENSEMBLE_WINS`.

The only robust constructive gain is the generic constrained dynamic KEEP pool
on exploratory OpenBMI: BA `0.850962`, Delta
`+0.452 pp`, paired subject-bootstrap 95% CI
`[+0.202, +0.721] pp`, with all five folds
positive. ACTION loses `-0.452 pp`
relative to that model. PERSIST then changes BA by
`-0.087 pp` with CI
`[-0.298, +0.125] pp`
and worsens rather than improves the measured harm/worst-subject endpoints.

WBCIC development does not confirm the architecture. The corrected strongest
static reference is the five-expert probability mean (BA
`0.803626`). Direct
masked-pool transfer is `-0.170 pp`.
The best adapted generic linear stack reaches BA `0.806789`
and `+0.316 pp`, but its CI
`[-0.184, +0.817] pp` crosses zero. Therefore
`READY_FOR_OUTER_FREEZE` is not justified.

## Final development table

| dataset | table_role | method_id | mean_subject_BA | Delta_BA_vs_B_STRONG | CI95_L | CI95_U |
| --- | --- | --- | --- | --- | --- | --- |
| OpenBMI | Single model | B0_TARGET_KEEP_historical_pooled | 0.8233173076923076 | -0.02312499999999995 | nan | nan |
| OpenBMI | Static strong ensemble | A0_STATIC_B_STRONG | 0.8464423076923075 | 0.0 | 0.0 | 0.0 |
| OpenBMI | Dynamic KEEP-only / best final | A1_DYNAMIC_KEEP_FINAL | 0.8509615384615383 | 0.0045192307692307 | 0.0020192307692307 | 0.0072115384615384 |
| OpenBMI | KEEP+ACTION without PERSIST | A2_KEEP_ACTION_NO_PERSIST | 0.8464423076923077 | -6.405132834375904e-18 | -0.0038461538461538 | 0.0037499999999999 |
| OpenBMI | KEEP+ACTION+PERSIST | A3_KEEP_ACTION_PERSIST | 0.845576923076923 | -0.0008653846153846 | -0.0047115384615384 | 0.0026923076923076 |
| OpenBMI | Best generic stacking control | M1_DYNAMIC_KEEP_LINEAR | 0.848846153846154 | 0.0024038461538461 | -0.0005769230769231 | 0.0056730769230769 |
| OpenBMI | Oracle KEEP-only | ORACLE_KEEP_ONLY | 0.9230769230769231 | 0.07663461538461558 | nan | nan |
| OpenBMI | Oracle KEEP+ACTION | ORACLE_KEEP_ACTION | 0.9534615384615386 | 0.10701923076923103 | nan | nan |
| WBCIC-development | Single model | W0_EEGNET_STABLE | 0.794354982754373 | -0.009270756343927 | -0.0169590724316334 | -0.0009780734170978 |
| WBCIC-development | Static strong ensemble | W0_B_STRONG_PROBABILITY_MEAN | 0.8036257390983 | 0.0 | 0.0 | 0.0 |
| WBCIC-development | Direct transferred Dynamic KEEP | W1_MASKED_POOL_SHRUNK_THR | 0.8019273527962552 | -0.0016983863020448 | -0.0073170731707316 | 0.0034146341463414 |
| WBCIC-development | Best generic stacking / development candidate | W1_RAW_LINEAR | 0.8067887718649914 | 0.0031630327666913 | -0.0018369672333086 | 0.0081733955407735 |
| WBCIC-development | Oracle KEEP-only | WBCIC_ORACLE_KEEP_ONLY | 0.9713414634146342 | 0.1677157243163342 | nan | nan |
| WBCIC-development | KEEP+ACTION / PERSIST | NOT_AVAILABLE_NO_LEGAL_ACTION_EXPERT | nan | nan | nan | nan |

## Answers to the required questions

1. **Strongest static ensemble.** OpenBMI: `B6_ALL_RUN_LOGIT_MEAN`, BA
   `0.846442`. WBCIC-dev: the five competent
   experts' probability mean, BA `0.803626`.
2. **Best final architecture.** A positive availability-normalized pool over
   frozen KEEP logits, with L2-shrunk log-weights, one scale/bias, and a narrow
   inner-calibrated threshold. It is a discovery model, not an outer-ready model.
3. **Major families tested.** Eight: threshold calibration, generic linear
   stacking, shallow HGB, anchored bounded residuals, positive logit pooling,
   positive probability pooling, contextual pooling, and DeepSets.
4. **Failures.** Trees and DeepSets lacked stable grouped gain; probability
   pooling did not beat its static WBCIC counterpart; residual corrections
   remained unstable; ACTION was harm-dominated; flat PERSIST inputs added no
   independent value; direct cross-benchmark transfer failed.
5. **Final Delta BA.** OpenBMI `+0.452 pp`.
   WBCIC best adapted exploratory candidate `+0.316 pp`.
6. **Grouped CI.** OpenBMI
   `[+0.202, +0.721] pp`; WBCIC adapted
   `[-0.184, +0.817] pp`.
7. **Stability.** OpenBMI 5/5 folds positive, 50.0% subjects positive and 82.7%
   nonnegative, worst subject `-1.5 pp`. WBCIC 4/5 folds positive but only 43.9%
   subjects positive, 65.9% nonnegative, worst subject `-3.5 pp`.
8. **Dynamic KEEP value.** Yes on OpenBMI (`+0.452 pp`, LCB > 0); not confirmed
   on WBCIC.
9. **ACTION value.** No. A2-A1 is
   `-0.452 pp` with a fully negative CI.
10. **PERSIST raw value.** No. A3-A2 is
    `-0.087 pp`; CI crosses zero.
11. **PERSIST safety/robustness.** No. It increases harm rate by
    `+1.837 pp` and worsens
    the worst-subject endpoint by `-0.500 pp`.
12. **PERSIST features.** No category has reliable incremental value. Decision
    dependence and persistence-only increments are small with CIs crossing zero;
    protected inputs have a negative point estimate.
13. **Selected experts.** The winning model uses KEEP experts only. Fold-level
    weights are in `diagnostics/EXPERT_USAGE.csv`; no ACTION expert is selected.
14. **ERASE necessary.** No. It is excluded; WBCIC's existing development audit
    also found no actionable harmful block and large harm from erasing protected structure.
15. **Soft residual vs hard switch.** No evidence of superiority. Bounded
    residual correction remains below B_STRONG, and the V3 hard policies also fail.
16. **Generic stacker.** Yes: the final gain is generic dynamic ensemble
    aggregation, not a PERSIST-aware method. On WBCIC, generic linear stacking
    is also the best point estimate, but not statistically robust.
17. **Frozen representations.** Not evaluated as a final gate. Logit models had
    a real OpenBMI signal, while compatible representations for the full WBCIC
    expert roster were unavailable; adding a single-backbone representation
    would confound architecture and capacity.
18. **Transfer.** Direct transfer fails (`-0.170 pp`).
    Only benchmark-adapted generic stacking has positive point estimates on both.
19. **Best OpenBMI development performance.** BA `0.850962`.
20. **Best WBCIC development performance.** BA `0.806789`,
    exploratory and non-robust.
21. **Gain vs original single model.** OpenBMI final vs historical pooled B0:
    `+2.764 pp`. WBCIC adapted
    model vs EEGNet_STABLE: `+1.243 pp`.
22. **Gain vs B_STRONG.** OpenBMI `+0.452 pp`; WBCIC adapted `+0.316 pp` with
    CI crossing zero.
23. **Gain decomposition.** OpenBMI single-to-static: approximately `+2.313 pp`;
    static-to-dynamic KEEP: `+0.452 pp`; ACTION beyond dynamic KEEP: `-0.452 pp`;
    PERSIST beyond KEEP+ACTION: `-0.087 pp`.
24. **Outer evaluation justified?** No. OpenBMI is exploratory, direct WBCIC
    transfer is negative, and the adapted WBCIC candidate has LCB < 0.
25. **What must be frozen before outer.** The WBCIC expert checkpoint roster and
    hashes, probability-mean B_STRONG, one model family (not several candidates),
    preprocessing, exact C/regularization, a single calibration/threshold rule,
    seed, legality hashes, and evaluation script. V4 intentionally does not make
    that outer authorization.

## Scientific interpretation

The oracle gaps remain large (`+7.663 pp`
for KEEP-only and `+10.702 pp`
for the complete menu), but prospective models recover only a small fraction.
This supports a selection-bottleneck conclusion. It does not support the claim
that PERSIST diagnostics currently improve prediction.

## Legality

- OpenBMI is exploratory only.
- WBCIC uses 41 authorized development subjects and S3 only.
- Sealed outer IDs, labels, logits, metadata, and results were never loaded.
- `OUTER_TEST_USED=false`.
