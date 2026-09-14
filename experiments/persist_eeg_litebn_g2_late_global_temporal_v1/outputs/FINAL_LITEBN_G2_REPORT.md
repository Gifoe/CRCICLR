# LiteBN-G2 final report

Terminal: `SSVEP_BENCHMARK_FAIL`.

| Task | Benchmark best | Matched LiteBN | LiteBN-G2 | Delta vs B0 pp | Delta vs benchmark pp | Selected epochs | Epoch0 fallback count | Mean selected normalized attention entropy | Entropy<0.97 folds | PASS |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| OpenBMI_SSVEP | 0.9234 | 0.9070 | 0.9123 | 0.529 | -1.111 | 34;26;32;9;0 | 1 | 0.9472 | 2 | False |
| OpenBMI_ERP | 0.8557 | — | — | — | — | nan | — | — | — | False |
| OpenBMI_MI | 0.7555 | — | — | — | — | nan | — | — | — | False |
| WBCIC_MI | 0.7918 | — | — | — | — | nan | — | — | — | False |

## Required answers

1. Epoch0 exactly reproduced matched LiteBN: `True`.
2. Stem and backend1 parameters were bitwise unchanged: `True`.
3. All BatchNorm states and affine parameters were unchanged and eval-locked: `True`.
4. Late attention increased SSVEP relative to early LiteBN-G (0.9084285714): `True` (0.9122857142857144).
5. LiteBN-G2 exceeded matched LiteBN: `True`.
6. LiteBN-G2 exceeded 92.34%: `False` (0.9122857142857144).
7. Selected attention became more selective than prior near-uniform attention: `True` (mean normalized entropy=0.947247; selective folds=2/5).
8. Checkpoint selection selected epoch0 in 1/5 executed folds.
9. |alpha_attn| became materially larger than early LiteBN-G maximum 0.00373122: `True` (maximum=0.0813531).
10. Activation was mainly associated with: `both` (mean alpha_attn=-0.0288009; mean alpha_ffn=-0.0267499).
11. If SSVEP failed, the experiment terminated immediately: `True`.
12. If SSVEP passed, ERP/MI/WBCIC passed fixed gates: `False`.
13. Multiseed ran and all four means exceeded benchmarks: `False` (not permitted after a seed0 failure).
14. Late high-level global attention is justified as a final single-model direction: `False`.

All opened benchmark resources are labeled EXPOSED_DEVELOPMENT_BENCHMARK. NEW_SEALED_TEST_ACCESSED = NO.
