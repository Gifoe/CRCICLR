# LiteBN CE versus CE+PRD, seeds 0, 1, 2

PRD lambda=0.25 was reused without tuning. Fold and seed repetitions were averaged within each biological subject and session; WS-BA is the minimum of those session means. The 20,000-draw bootstrap resamples biological subjects.

OpenBMI V8 is an internal-heldout diagnostic cohort. The WBCIC true-outer cohort has already been accessed for prior ablation diagnostics and is therefore an exposed benchmark, not untouched final confirmation.

| Task | Condition | Alignment | Future BA | Macro-F1 | WS-BA |
|---|---|---:|---:|---:|---:|
| OpenBMI_MI | CE | 0.7892 | 0.7061 | 0.6980 | 0.6708 |
| OpenBMI_MI | CE_PLUS_PRD | 0.8945 | 0.7032 | 0.6940 | 0.6686 |
| WBCIC_MI | CE | 0.8018 | 0.7640 | 0.7592 | 0.6884 |
| WBCIC_MI | CE_PLUS_PRD | 0.7997 | 0.7471 | 0.7386 | 0.6732 |

| Task | Delta alignment [95% CI] | Delta BA pp [95% CI] | Delta Macro-F1 pp [95% CI] | Delta WS-BA pp [95% CI] |
|---|---:|---:|---:|---:|
| OpenBMI_MI | +0.1054 [+0.0722, +0.1391] | -0.295 [-0.686, +0.129] | -0.404 [-0.851, +0.078] | -0.214 [-0.657, +0.290] |
| WBCIC_MI | -0.0021 [-0.0401, +0.0278] | -1.693 [-2.507, -0.787] | -2.054 [-2.950, -1.071] | -1.520 [-2.297, -0.747] |
