# Final report

Terminal state: `SOURCE_ONLY_PERSIST_NOT_SUPPORTED`.

## Direct answers

1. **PUD source-only:** BA = **0.756500**, Macro-F1 = **0.753981**.
2. **Versus Vanilla EEGNet:** Vanilla BA = 0.786167; ΔBA = -0.029667 (-2.967 pp), 95% subject-bootstrap CI [-0.043417, -0.016998].
3. **Versus capacity-matched dual source control:** control BA = 0.777667; ΔBA = -0.021167, 95% CI [-0.029417, -0.013250].
4. **Versus Strong EEGNet:** Strong BA = 0.791500; ΔBA = -0.035000, 95% CI [-0.047250, -0.023583].
5. **Fold consistency versus Vanilla:** 0/5 positive.
6. **Seed consistency versus Vanilla:** 0/3 positive.
7. **Subject wins versus Vanilla:** 8/40 improved, 32/40 harmed, 0/40 tied.
8. **Target adaptation effect:** after-adaptation BA = 0.763917; after − source ΔBA = +0.007417, 95% CI [+0.003167, +0.011833]. Target adaptation helped the frozen source representation on average.
9. **Protected branch consequence:** erasure harm = +0.135583, 95% CI [+0.117083, +0.155417]; functional-teacher correlation = 0.820923.
10. **Interpretation:** `SOURCE_ONLY_PERSIST_NOT_SUPPORTED`.
11. **OpenBMI 14-subject internal holdout accessed? NO.**
12. **WBCIC sealed outer accessed? NO.**

## Scientific conclusion

PUD supervision did not improve future-session generalization over Vanilla EEGNet. This is not a marginally positive result and does not justify another constructive model search.

The protected pathway's erasure result and the end-to-end generalization result are separate facts; task consequence is not evidence of generalization benefit. The diagnostic is limited to the frozen OpenBMI V8_SEARCH protocol and cannot establish a broader cross-dataset claim.

## Integrity

B0/B1 replay passed before PUD evaluation. PUD source BA also matched the authoritative pre-adaptation BA stored independently in both A6 and A10 rows (maximum absolute error 1.110e-16). No checkpoint, parameter, BN buffer, or LayerNorm buffer changed. No training or adaptation was run.
