# PERSIST-EEG V5 scientific report

## Decision

Terminal state: `V5_DEVELOPMENT_TARGET_REACHED` and `READY_FOR_OUTER_FREEZE`.

The selected development candidate is **CS-LGS: Cross-Session Local-Geometry Stack**. On the authorized 41-subject WBCIC S1/S2->S3 development protocol it reached BA **0.817782**, a gain of **1.099 pp** over `W1_RAW_LINEAR` (paired subject bootstrap 95% CI **[0.195, 2.028] pp**). All five folds improved. The sealed outer cohort was not enumerated, loaded, or evaluated.

This is an exploratory development result after repeated hypothesis iteration. It is not an outer-confirmed generalization claim.

## Required answers

1. **Strongest legal baselines.** OpenBMI: `A1_DYNAMIC_KEEP_FINAL`, BA 0.850962. WBCIC development: `W1_RAW_LINEAR`, BA 0.806789. Static references were 0.846442 and 0.803626 respectively.
2. **Baseline evolution.** No stronger non-adaptive baseline was created. Stronger V5 rows all use target-subject S1/S2 adaptation and are reported as candidate ablations, not silently substituted baselines.
3. **Oracle headroom.** It is concentrated in WBCIC 3-2 disagreements: 2,092 trials, baseline BA about 0.622, with 811 majority-wrong/minority-correct trials.
4. **Stable complementarity.** EEGNet Stable was strongest (BA about 0.794); EEGNet Standard and DeepConvNet were weaker but useful in a stack. TeCh and especially EEGConformer were not competent standalone experts and inflated oracle headroom through occasional guesses.
5. **Disagreement state alone.** No. Output-only selectors were negative or negligible.
6. **EEG context.** Fold-compatible frozen context added only +0.148 pp with a CI crossing zero. The history-derived CSP score added 0.171 pp over `M12_SIMPLE_ALL_REFIT4`; its direct control BA was only 0.580, so it is treated as context, not a strong expert.
7. **Cross-session reliability.** Scalar S1/S2 expert reliability failed. Rich subject-local S1/S2 geometry was useful.
8. **Best target.** Direct subject-balanced robust stacking outperformed expert-correctness BCE, pairwise ranking, local-error utility, and kNN competence.
9. **Best aggregation.** A conservative anchor with a fixed non-unanimous gate. Hard selection, generic soft weighting, ranking, and correlation-aware pools were worse.
10. **Final Delta BA.** +1.099 pp versus `W1_RAW_LINEAR`.
11. **Subject-bootstrap CI.** [0.195, 2.028] pp.
12. **Positive folds.** 5/5.
13. **Subject stability.** 63.4% positive and 70.7% nonnegative.
14. **Oracle recovery.** 35.7% of available rescue trials, using the experiment's rescue-count definition.
15. **PERSIST raw BA.** No supported increment. The V4 OpenBMI PERSIST increment was negative and its CI crossed zero.
16. **PERSIST safety.** Yes, narrowly: the frozen WBCIC audit marked P01_04 protected and authorized no action. PERSIST is retained as a veto, not claimed as a performance feature.
17. **ACTION experts.** No. The WBCIC actionability audit found no block passing H1-H5; OpenBMI ACTION was negative.
18. **Full ERASE.** Not necessary and not authorized.
19. **Cross-benchmark behavior.** WBCIC met the development target. OpenBMI has fewer than two prior sessions, so the method fail-closed to A1 exactly: Delta BA 0.000 pp (non-degrading, not improving).
20. **Target-subject adaptation.** Yes: labeled S1/S2 only. No held-out target S3 label or target-batch statistic entered prediction.
21. **Outer access.** No. `OUTER_TEST_USED=false` throughout.
22. **Distinct families tried.** At least ten: scalar reliability, output direct, handcrafted context, shared frozen context, error rescue, kNN, multi-label correctness, pairwise ranking, covariance aggregation, subject-local heads, and CSP-context stacking.
23. **Failure progression.** Each failed family is retained in `outputs/research_log/`; the progression moved from zero-shot trial competence to legal history-conditioned subject geometry, then to a fixed low-variance stack.
24. **Target reached.** Yes on WBCIC development under the frozen non-adaptive baseline definition.
25. **Ready to freeze.** Yes as a later outer-evaluation candidate. This does not authorize opening outer data.

## Limitations

- The final gain is only 1.099 pp and the worst subject changed by -6.0 pp; it is not uniformly beneficial.
- CSP is a weak standalone classifier. Its small incremental value may not replicate; the untouched outer cohort is essential.
- Repeated development iterations make the estimate exploratory despite nested grouped evaluation.
- The OpenBMI result is a safety fallback, not evidence that the new history-conditioned mechanism transfers to a two-session dataset.
- PERSIST did not produce predictive gain in the selected model.
