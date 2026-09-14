# LiteBN-TFFormer final report

Terminal: `SSVEP_BENCHMARK_FAIL`.

SSVEP matched LiteBN=0.907000; TFFormer=0.914286; G2 reference=0.9122857143; benchmark=0.9234.

## Required answers

1. Outperformed matched LiteBN: `True`.
2. Outperformed G2: `True`.
3. Exceeded 0.9234: `False`.
4. B0 fallback folds: `0`.
5. Spectral branch had non-zero first-step gradients: `True`.
6. TF cross-attention became active: `True`.
7. Per-scale masses are in SPECTRAL_BRANCH_DIAGNOSTICS.csv.
8. Global blocks remained active: `True`.
9. Local Conv mixer became active: `True`.
10. Execution stopped after SSVEP: `True`.
11–15. ERP, MI, WBCIC, and multiseed were not run after the SSVEP failure.
16. TFFormer is justified as final candidate: `False`.

All evaluation resources are EXPOSED_DEVELOPMENT_BENCHMARK; NEW_SEALED_TEST_ACCESSED = NO.
