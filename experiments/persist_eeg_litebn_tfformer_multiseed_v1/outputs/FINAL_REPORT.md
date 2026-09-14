# TFFormer four-task, three-seed diagnostic

## Per-seed internal-heldout results

| seed | task | TFFormer | LiteBN | delta_vs_LiteBN_pp | benchmark | delta_vs_benchmark_pp | PASS |
|---|---|---|---|---|---|---|---|
| 0 | OpenBMI_ERP | 0.849805 | 0.846545 | 0.325974 | 0.855700 | -0.589481 | False |
| 0 | OpenBMI_MI | 0.741571 | 0.744857 | -0.328571 | 0.755500 | -1.392857 | False |
| 0 | OpenBMI_SSVEP | 0.914286 | 0.907000 | 0.728571 | 0.923400 | -0.911429 | False |
| 0 | WBCIC_MI | 0.795800 | 0.792200 | 0.360000 | 0.791800 | 0.400000 | True |
| 1 | OpenBMI_ERP | 0.846030 | 0.844506 | 0.152381 | 0.855700 | -0.966970 | False |
| 1 | OpenBMI_MI | 0.753429 | 0.750714 | 0.271429 | 0.755500 | -0.207143 | False |
| 1 | OpenBMI_SSVEP | 0.911000 | 0.911286 | -0.028571 | 0.923400 | -1.240000 | False |
| 1 | WBCIC_MI | 0.798100 | 0.796500 | 0.160000 | 0.791800 | 0.630000 | True |
| 2 | OpenBMI_ERP | 0.852532 | 0.852848 | -0.031602 | 0.855700 | -0.316753 | False |
| 2 | OpenBMI_MI | 0.749714 | 0.751571 | -0.185714 | 0.755500 | -0.578571 | False |
| 2 | OpenBMI_SSVEP | 0.913429 | 0.912429 | 0.100000 | 0.923400 | -0.997143 | False |
| 2 | WBCIC_MI | 0.795100 | 0.793300 | 0.180000 | 0.791800 | 0.330000 | True |

## Three-seed task aggregates

| task | TFFormer_mean | TFFormer_std | LiteBN_mean | LiteBN_std | paired_delta_mean_pp | paired_delta_std_pp | benchmark | delta_mean_vs_benchmark_pp | benchmark_pass_seeds |
|---|---|---|---|---|---|---|---|---|---|
| OpenBMI_SSVEP | 0.912905 | 0.001704 | 0.910238 | 0.002862 | 0.266667 | 0.405154 | 0.923400 | -1.049524 | 0 |
| OpenBMI_ERP | 0.849456 | 0.003265 | 0.847967 | 0.004349 | 0.148918 | 0.178813 | 0.855700 | -0.624401 | 0 |
| OpenBMI_MI | 0.748238 | 0.006065 | 0.749048 | 0.003654 | -0.080952 | 0.313419 | 0.755500 | -0.726190 | 0 |
| WBCIC_MI | 0.796333 | 0.001570 | 0.794000 | 0.002234 | 0.233333 | 0.110151 | 0.791800 | 0.453333 | 3 |

## Equal-task aggregate by seed

| seed | equal_task_TFFormer_mean | equal_task_LiteBN_mean | equal_task_delta_pp | benchmark_pass_tasks |
|---|---|---|---|---|
| 0 | 0.825366 | 0.822651 | 0.271494 | 1 |
| 1 | 0.827140 | 0.825752 | 0.138810 | 1 |
| 2 | 0.827694 | 0.827537 | 0.015671 | 1 |

Equal-task mean paired delta versus matched LiteBN: +0.141991 pp.
Task means above hard benchmark: 1/4.

## Decision

`STOP_MODEL`

Only WBCIC MI exceeds its hard benchmark in the three-seed mean; OpenBMI SSVEP, ERP, and MI remain below benchmark. OpenBMI MI also has a negative mean paired delta versus matched LiteBN.

Protocol limitation: Seed0 SSVEP used the historical full-60-epoch schedule while seed1/2 used user-requested min10/patience8 early stopping. Treat this as an internal multiseed diagnostic, not a strict uniform-protocol estimate.

These are internal-heldout diagnostics. No new sealed test was accessed.
