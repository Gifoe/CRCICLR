# LiteBN-G final report

Terminal: `SSVEP_BENCHMARK_FAIL`.

| Task | Benchmark best | Benchmark model | Matched LiteBN | LiteBN-G | Delta vs LiteBN pp | Delta vs benchmark pp | Selected epochs | Epoch0 fallback count | PASS |
|---|---:|---|---:|---:|---:|---:|---|---:|---|
| OpenBMI_SSVEP | 0.9234 | TCFormer | 0.9070 | 0.9084 | 0.143 | -1.497 | 12;0;21;8;0 | 2 | False |
| OpenBMI_ERP | 0.8557 | TCFormer | — | — | — | — | nan | — | False |
| OpenBMI_MI | 0.7555 | TCFormer | — | — | — | — | nan | — | False |
| WBCIC_MI | 0.7918 | EEGNet | — | — | — | — | nan | — | False |

## Required answers

1. Epoch0 reproduced exact historical LiteBN: `True`.
2. All early-stem weights were bitwise unchanged: `True`.
3. All BatchNorm parameters/buffers were bitwise unchanged and eval-locked: `True`.
4. Seed0 SSVEP exceeded 92.34%: `False` (0.9084285714285716).
5. If SSVEP failed, immediate termination occurred: `True`.
6. ERP exceeded 85.57%: `False` (not reached).
7. OpenBMI MI exceeded 75.55%: `False` (not reached).
8. WBCIC exceeded 79.18%: `False` (not reached).
9. WBCIC preserved or exceeded matched LiteBN: `False`.
10. Maximum selected |alpha_attn|=0.00373122; |alpha_ffn|=0.0468263.
11. Checkpoint selection fell back to epoch0 in 2/5 executed folds.
12. The global block became active: `True`.
13. Multiseed ran: `False` (only permitted after all seed0 task gates pass).
14. LiteBN-G is justified as final single-model candidate: `False`.

All opened benchmark resources are labeled EXPOSED_DEVELOPMENT_BENCHMARK. NEW_SEALED_TEST_ACCESSED = NO.
