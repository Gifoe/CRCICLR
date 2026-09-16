# Baseline metrics closure — amended final true-outer scope

Status: **AMENDED-SCOPE ANALYSIS COMPLETE; full 6×4 three-seed table remains incomplete.**

## A. Completion audit

ModernTCN, Medformer, EEGNet, CBraMod and TeCh each have 15/15 verified frozen checkpoints in all four tasks. LiteBN is 15/15 for OpenBMI MI and ERP, 13/15 for SSVEP and 5/15 for WBCIC MI. The last two cells are not presented as full three-seed results. No training or checkpoint reselection occurred.

## B–D. Unified retained-baseline benchmark

See `MAIN_PREDICTIVE_TABLE.md` and `FINAL_FULLMODEL_METRICS.csv`. The table uses 14 OpenBMI final heldout subjects and 10 WBCIC final true-outer subjects. The originally supplied WBCIC numerical targets belonged to the disjoint V8 internal cohort; they were retired as final-outer targets by the approved amendment.

| Method | Task | BA | Macro-F1 | WS-BA | Completion |
|---|---|---:|---:|---:|---|
| EEGNet | OpenBMI_MI | 69.84 | 69.24 | 67.16 | 15/15 |
| EEGNet | OpenBMI_ERP | 84.68 | 80.43 | 81.14 | 15/15 |
| EEGNet | OpenBMI_SSVEP | 89.02 | 88.93 | 87.29 | 15/15 |
| EEGNet | WBCIC_MI | 79.18 | 78.74 | 70.65 | 15/15 |
| CBraMod | OpenBMI_MI | 50.68 | 48.18 | 49.70 | 15/15 |
| CBraMod | OpenBMI_ERP | 74.74 | 69.22 | 72.89 | 15/15 |
| CBraMod | OpenBMI_SSVEP | 85.03 | 84.90 | 83.10 | 15/15 |
| CBraMod | WBCIC_MI | 75.45 | 75.27 | 67.32 | 15/15 |
| TeCh | OpenBMI_MI | 71.18 | 69.89 | 67.37 | 15/15 |
| TeCh | OpenBMI_ERP | 80.46 | 75.07 | 76.87 | 15/15 |
| TeCh | OpenBMI_SSVEP | 82.35 | 82.20 | 80.08 | 15/15 |
| TeCh | WBCIC_MI | 74.37 | 74.10 | 66.49 | 15/15 |
| ModernTCN | OpenBMI_MI | 61.70 | 61.31 | 57.55 | 15/15 |
| ModernTCN | OpenBMI_ERP | 79.81 | 76.84 | 76.64 | 15/15 |
| ModernTCN | OpenBMI_SSVEP | 84.42 | 84.24 | 82.64 | 15/15 |
| ModernTCN | WBCIC_MI | 71.47 | 71.33 | 64.72 | 15/15 |
| Medformer | OpenBMI_MI | 67.16 | 66.64 | 63.26 | 15/15 |
| Medformer | OpenBMI_ERP | 80.07 | 78.90 | 76.62 | 15/15 |
| Medformer | OpenBMI_SSVEP | 88.61 | 88.47 | 87.07 | 15/15 |
| Medformer | WBCIC_MI | 75.81 | 75.53 | 68.92 | 15/15 |
| LiteBN | OpenBMI_MI | 72.13 | 70.93 | 69.08 | 15/15 |
| LiteBN | OpenBMI_ERP | 85.04 | 80.71 | 81.82 | 15/15 |
| LiteBN | OpenBMI_SSVEP | INCOMPLETE | INCOMPLETE | INCOMPLETE | INCOMPLETE |
| LiteBN | WBCIC_MI | INCOMPLETE | INCOMPLETE | INCOMPLETE | INCOMPLETE |

## B. ModernTCN frozen-session detail

| Task | Session | BA (%) | Macro-F1 (%) | WS-BA (%) | CSGD (pp) | Params | MACs (M) |
|---|---|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | S1 | 60.70 | 60.51 | 57.55 | -1.00 | 17391170 | 688.84 |
| OpenBMI_MI | S2 | 61.70 | 61.31 | 57.55 | -1.00 | 17391170 | 688.84 |
| OpenBMI_ERP | S1 | 80.14 | 77.09 | 76.64 | 0.33 | 17026114 | 173.42 |
| OpenBMI_ERP | S2 | 79.81 | 76.84 | 76.64 | 0.33 | 17026114 | 173.42 |
| OpenBMI_SSVEP | S1 | 88.19 | 88.04 | 82.64 | 3.77 | 17883204 | 689.34 |
| OpenBMI_SSVEP | S2 | 84.42 | 84.24 | 82.64 | 3.77 | 17883204 | 689.34 |
| WBCIC_MI | S1 | 71.68 | 71.50 | 64.72 | 0.25 | 15914050 | 629.67 |
| WBCIC_MI | S2 | 71.75 | 71.58 | 64.72 | 0.25 | 15914050 | 629.67 |
| WBCIC_MI | S3 | 71.47 | 71.33 | 64.72 | 0.25 | 15914050 | 629.67 |

## C. Medformer frozen-session detail

| Task | Session | BA (%) | Macro-F1 (%) | WS-BA (%) | CSGD (pp) | Params | MACs (M) |
|---|---|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | S1 | 66.47 | 66.10 | 63.26 | -0.69 | 3890178 | 587.91 |
| OpenBMI_MI | S2 | 67.16 | 66.64 | 63.26 | -0.69 | 3890178 | 587.91 |
| OpenBMI_ERP | S1 | 80.74 | 79.19 | 76.62 | 0.66 | 3781890 | 133.79 |
| OpenBMI_ERP | S2 | 80.07 | 78.90 | 76.62 | 0.66 | 3781890 | 133.79 |
| OpenBMI_SSVEP | S1 | 92.20 | 92.11 | 87.07 | 3.60 | 4035332 | 588.05 |
| OpenBMI_SSVEP | S2 | 88.61 | 88.47 | 87.07 | 3.60 | 4035332 | 588.05 |
| WBCIC_MI | S1 | 76.00 | 75.50 | 68.92 | 0.53 | 3853314 | 584.81 |
| WBCIC_MI | S2 | 76.68 | 76.40 | 68.92 | 0.53 | 3853314 | 584.81 |
| WBCIC_MI | S3 | 75.81 | 75.53 | 68.92 | 0.53 | 3853314 | 584.81 |

## E. Representation-only PSWA extension

Published EEGNet/TeCh rows reproduce exactly (8/8). The exact published recovery script, frozen PEEH assignments, and deterministic original random controls were used. Empty Protected assignments are excluded as undefined, not converted to zero.

| Model | Task | Coverage | PSWA pp [95% subject CI] |
|---|---|---:|---:|
| ModernTCN | OpenBMI_MI | 1/5 | -0.10 [-2.27, 2.03] |
| ModernTCN | OpenBMI_ERP | 5/5 | 13.04 [10.63, 15.46] |
| ModernTCN | OpenBMI_SSVEP | 5/5 | 22.46 [15.49, 28.99] |
| ModernTCN | WBCIC_MI | 3/5 | 9.17 [4.57, 13.94] |
| Medformer | OpenBMI_MI | 5/5 | 5.45 [3.66, 7.26] |
| Medformer | OpenBMI_ERP | 5/5 | 8.13 [6.97, 9.22] |
| Medformer | OpenBMI_SSVEP | 5/5 | 28.37 [24.21, 31.85] |
| Medformer | WBCIC_MI | 5/5 | 12.04 [6.96, 17.56] |

## F. Statistical uncertainty

20,000 deterministic percentile bootstrap draws resample biological subjects only. Checkpoint/session replicates are averaged within a subject first. Paired LiteBN contrasts are reported only for complete identical-cohort cells; a CI crossing zero is not equivalence.

## G. Leakage and regression audit

OpenBMI future-session frozen re-evaluation matches all six pre-existing numerical targets within 1e-8. For WBCIC, an independent frozen seed0 rerun matches 100 committed true-outer subject/fold rows; there is no independent published 15-checkpoint true-outer target, so this remains a documented scope limitation. All per-session rows retain checkpoint and normalizer hashes. WBCIC file S0/S1/S2 maps to paper S1/S2/S3.

Native-batch OpenBMI per-subject future rows match the published evaluator exactly (1260/1260). WBCIC seed0 comparison to the old batch-32 CSGD artifact is exact for 100/100 rows; maximum BA difference is 0.00000000.

ModernTCN/Medformer were re-evaluated in a preserved, separate runtime using the original frozen evaluators' batch size 128. A single Medformer ERP subject/checkpoint prediction differed at CSGD batch 32; batch 128 reproduces the published OpenBMI target exactly. The batch-32 runtime was retained for audit.

Source commits: seven-backbone true-outer `097c7f5016fb6690800c57452eb1ec9657a2fc7b`; ModernTCN `a9c6a6a4ecd363242a5cd7a0608abe3ae7d73b79`; Medformer `68ef63a182ee250b0b6865182ed94fcebec6e89d`; cross-backbone CSGD `4020f0bb0a24d5ef70bb9ba0e2116b7b0f257ebb`; PEEH `bd2cc4346041d16f544963703e02bf93e262fbd3`; published PSWA `3fd6aad50c08144c95a58454ca76d58f21aedc25`. Per-cell hashes are in the frozen inference and PSWA locks.

## H. Manuscript recommendation and remaining gaps

Use future BA, macro-F1, WS-BA and Params in the main predictive table. Put session-wise BA/F1, CSGD, subject bootstrap CIs, paired contrasts and MAC audit in appendix. PSWA is a representation diagnostic, not a substitute for full-model WS-BA. Do not claim a complete six-model four-task three-seed table: LiteBN has two incomplete cells. Do not report CBraMod MACs as numeric under the current profiler: FFT is uncounted.

PSWA extension cells: 40; nonempty: 34; empty: 6.
