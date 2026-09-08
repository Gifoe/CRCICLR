# NRRF BatchNorm Buffer Intervention Audit

| Metric | OpenBMI JOINT-CE | OpenBMI NRRF | WBCIC JOINT-CE | WBCIC NRRF |
|---|---:|---:|---:|---:|
| Current BA | 0.7995 | 0.7973 | 0.7463 | 0.7382 |
| Restore-source-BN BA | 0.8010 | 0.8017 | 0.7937 | 0.7901 |
| BN recovery (pp) | +0.150 | +0.450 | +4.740 | +5.195 |
| Recovery fraction | 300.0% | 163.6% | 99.3% | 93.1% |
| Current harm ≤ −1pp | 17.5% | 15.0% | 58.1% | 61.3% |
| Restored harm ≤ −1pp | 12.5% | 12.5% | 25.8% | 25.8% |
| Improved folds | 3/5 | 4/5 | 4/5 | 3/5 |

1. Does source-BN restoration materially recover WBCIC? **YES**.
2. WBCIC recovery fractions: JOINT-CE 99.3%; NRRF-v1 93.1%
3. WBCIC fold0/fold3 changes (CURRENT delta, RESTORED delta, recovery pp):
   - JOINT-CE fold0: -9.500, -0.071, +9.429
   - JOINT-CE fold1: -1.417, -1.333, +0.083
   - JOINT-CE fold2: +0.917, +2.000, +1.083
   - JOINT-CE fold3: -9.896, +2.593, +12.489
   - JOINT-CE fold4: +2.667, +2.500, -0.167
   - NRRF-v1 fold0: -8.714, +0.286, +9.000
   - NRRF-v1 fold1: -1.583, -1.667, -0.083
   - NRRF-v1 fold2: +0.333, +1.333, +1.000
   - NRRF-v1 fold3: -13.583, +1.841, +15.424
   - NRRF-v1 fold4: +2.000, +2.000, +0.000
4. Harmful-subject tails are reported in `HARM_PROFILE.json`.
5. OpenBMI BN sensitivity: see main table.
6. Raw epoch20 secondary check: see `RAW_EPOCH20_SECONDARY.json`.
7. BatchNorm buffer drift major explanation? YES.
8. Final terminal: **BN_BUFFER_DRIFT_MAJOR_CONTRIBUTOR**.
9. Next action: Test a new stage-2 training protocol in which the LiteBN BatchNorm running buffers are frozen at their original carrier values throughout training.
