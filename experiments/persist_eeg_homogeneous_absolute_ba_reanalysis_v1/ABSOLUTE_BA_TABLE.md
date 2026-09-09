# Absolute subject-equal balanced accuracy

Primary fusion is LOGIT50. All values are percentage points; brackets are 95% subject-bootstrap CIs with 10,000 resamples and fixed seed 20260909.

| Block | EEGNet+EEGNet BA | LiteBN+LiteBN BA | EEGNet+LiteBN BA | EL minus EE pp | EL minus LL pp | EL minus HOMOAVG pp | n subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI MI | 73.257 [68.495, 78.805] | 75.919 [70.562, 81.586] | 75.586 [70.526, 81.086] | 2.329 [0.762, 3.919] | -0.333 [-1.064, 0.460] | 0.998 [0.495, 1.524] | 14 |
| OpenBMI ERP | 85.383 [81.031, 89.687] | 85.401 [81.006, 89.547] | 85.705 [81.287, 89.856] | 0.322 [0.039, 0.670] | 0.304 [-0.025, 0.599] | 0.313 [0.198, 0.437] | 14 |
| OpenBMI SSVEP | 90.695 [83.943, 96.219] | 92.124 [85.857, 97.014] | 91.979 [85.578, 96.964] | 1.283 [0.445, 2.467] | -0.145 [-0.910, 0.452] | 0.569 [0.279, 0.895] | 14 |
| WBCIC MI | 79.737 [69.613, 88.770] | 79.897 [69.930, 88.860] | 80.027 [70.033, 89.157] | 0.290 [-0.710, 1.260] | 0.130 [-0.515, 0.867] | 0.210 [0.030, 0.407] | 10 |

## Optional exactly matched single-model reference

The values below are reconstructed only from the same frozen homogeneous-pair ledger. Each of the three seeds appears twice per fold, producing 30 matched seed/fold values per subject; no number is copied from a different experiment.

| Block | EEGNet BA | LiteBN BA | n subjects |
|---|---:|---:|---:|
| OpenBMI MI | 72.286 [67.686, 77.643] | 74.905 [69.490, 80.552] | 14 |
| OpenBMI ERP | 84.816 [80.450, 89.126] | 84.797 [80.360, 89.110] | 14 |
| OpenBMI SSVEP | 89.076 [81.833, 95.233] | 91.024 [84.452, 96.214] | 14 |
| WBCIC MI | 79.363 [69.400, 88.433] | 79.400 [70.090, 88.014] | 10 |
