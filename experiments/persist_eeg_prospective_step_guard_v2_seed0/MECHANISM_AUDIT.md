# Mechanism audit

`results/STEP_GUARD_LOG_SUMMARY.csv` aggregates trigger rate, correction size and h before/after. `results/B_PROBE_HARM.csv` is a held-out B-probe test using trials disjoint from the guard batch.

| dataset   | candidate         | scope   |   kappa | guard_kind   |   n_steps |   trigger_rate |   mean_relative_correction |   median_relative_correction |   mean_correction_norm |   mean_h_before |   mean_h_after |   max_cap_violation |   mean_delta_task_norm |   max_cap_ratio |
|:----------|:------------------|:--------|--------:|:-------------|----------:|---------------:|---------------------------:|-----------------------------:|-----------------------:|----------------:|---------------:|--------------------:|-----------------------:|----------------:|
| OpenBMI   | CAP_ZERO_IDENTITY | ALL     |    0    | TRUE_GUARD   |      1705 |      0.456891  |                  0         |                            0 |            0           |    -3.08993e-05 |   -3.08993e-05 |         0           |             0.00134885 |        0        |
| OpenBMI   | ERM2_REFERENCE    | ALL     |    0    | TRUE_GUARD   |      1705 |      0.0205279 |                  0         |                            0 |            0           |    -0.00047409  |   -0.00047409  |         0           |             0.00139179 |        0        |
| OpenBMI   | P1_ALL_CAP05      | ALL     |    0.05 | TRUE_GUARD   |      1705 |      0.46217   |                  0.0136041 |                            0 |            1.82034e-05 |    -2.93474e-05 |   -8.66254e-05 |         0.000197292 |             0.00134873 |        1        |
| OpenBMI   | P2_ALL_CAP10      | ALL     |    0.1  | TRUE_GUARD   |      1705 |      0.461584  |                  0.0166312 |                            0 |            2.23517e-05 |    -2.9038e-05  |   -0.000100142 |         0.000355621 |             0.00134874 |        1        |
| OpenBMI   | P3_ALL_CAP20      | ALL     |    0.2  | TRUE_GUARD   |      1705 |      0.460997  |                  0.0170393 |                            0 |            2.29338e-05 |    -2.89932e-05 |   -0.000102193 |         0.000355621 |             0.00134873 |        0.860485 |
| OpenBMI   | P4_LATE_CAP05     | LATE    |    0.05 | TRUE_GUARD   |      1705 |      0.457478  |                  0.0202831 |                            0 |            2.63135e-05 |    -3.07073e-05 |   -4.9852e-05  |         0.000268596 |             0.00134914 |        1        |
| OpenBMI   | P5_LATE_CAP10     | LATE    |    0.1  | TRUE_GUARD   |      1705 |      0.459238  |                  0.0357038 |                            0 |            4.63477e-05 |    -3.05493e-05 |   -6.43649e-05 |         0.000537186 |             0.00134935 |        1        |
| OpenBMI   | P6_LATE_CAP20     | LATE    |    0.2  | TRUE_GUARD   |      1705 |      0.458651  |                  0.055629  |                            0 |            7.21462e-05 |    -3.0337e-05  |   -8.30855e-05 |         0.000763273 |             0.00134959 |        1        |
| OpenBMI   | TASK_ONLY_MATCHED | ALL     |    0    | TRUE_GUARD   |      1705 |      0.456891  |                  0         |                            0 |            0           |    -3.08993e-05 |   -3.08993e-05 |         0           |             0.00134885 |        0        |
| WBCIC     | CAP_ZERO_IDENTITY | ALL     |    0    | TRUE_GUARD   |      2580 |      0.409302  |                  0         |                            0 |            0           |    -3.18723e-05 |   -3.18723e-05 |         0           |             0.00140314 |        0        |
| WBCIC     | ERM2_REFERENCE    | ALL     |    0    | TRUE_GUARD   |      2580 |      0.0108527 |                  0         |                            0 |            0           |    -0.00030127  |   -0.00030127  |         0           |             0.00145067 |        0        |
| WBCIC     | P1_ALL_CAP05      | ALL     |    0.05 | TRUE_GUARD   |      2580 |      0.40969   |                  0.0115059 |                            0 |            1.60063e-05 |    -3.17146e-05 |   -5.65855e-05 |         0.000276959 |             0.00140332 |        1        |
| WBCIC     | P2_ALL_CAP10      | ALL     |    0.1  | TRUE_GUARD   |      2580 |      0.410853  |                  0.0136879 |                            0 |            1.91175e-05 |    -3.17198e-05 |   -6.15789e-05 |         0.000328852 |             0.00140351 |        1        |
| WBCIC     | P3_ALL_CAP20      | ALL     |    0.2  | TRUE_GUARD   |      2580 |      0.410853  |                  0.0140243 |                            0 |            1.9624e-05  |    -3.17371e-05 |   -6.24551e-05 |         0.000328852 |             0.00140356 |        1        |
| WBCIC     | P4_LATE_CAP05     | LATE    |    0.05 | TRUE_GUARD   |      2580 |      0.408915  |                  0.0162629 |                            0 |            2.17732e-05 |    -3.1803e-05  |   -4.52687e-05 |         0.000268578 |             0.00140328 |        1        |
| WBCIC     | P5_LATE_CAP10     | LATE    |    0.1  | TRUE_GUARD   |      2580 |      0.408527  |                  0.0259238 |                            0 |            3.48815e-05 |    -3.1761e-05  |   -5.3415e-05  |         0.000537155 |             0.00140338 |        1        |
| WBCIC     | P6_LATE_CAP20     | LATE    |    0.2  | TRUE_GUARD   |      2580 |      0.40814   |                  0.0339109 |                            0 |            4.57813e-05 |    -3.17475e-05 |   -6.03309e-05 |         0.000874266 |             0.0014035  |        1        |
| WBCIC     | TASK_ONLY_MATCHED | ALL     |    0    | TRUE_GUARD   |      2580 |      0.409302  |                  0         |                            0 |            0           |    -3.18723e-05 |   -3.18723e-05 |         0           |             0.00140314 |        0        |

## B-probe

| dataset   | candidate         | scope   |   kappa |   epoch | guard_kind   |   n_probe |   harm_frequency |   mean_positive_harm |   mean_delta | probe_disjoint   |
|:----------|:------------------|:--------|--------:|--------:|:-------------|----------:|-----------------:|---------------------:|-------------:|:-----------------|
| OpenBMI   | CAP_ZERO_IDENTITY | ALL     |    0    |       1 | TRUE_GUARD   |        15 |         0.4      |          8.24342e-05 | -0.000120982 | True             |
| OpenBMI   | CAP_ZERO_IDENTITY | ALL     |    0    |       2 | TRUE_GUARD   |        15 |         0.6      |          0.000169882 |  0.000126857 | True             |
| OpenBMI   | CAP_ZERO_IDENTITY | ALL     |    0    |       3 | TRUE_GUARD   |        20 |         0.5      |          4.59678e-05 | -5.25244e-05 | True             |
| OpenBMI   | CAP_ZERO_IDENTITY | ALL     |    0    |       4 | TRUE_GUARD   |        15 |         0.533333 |          0.000124417 |  1.82569e-05 | True             |
| OpenBMI   | CAP_ZERO_IDENTITY | ALL     |    0    |       5 | TRUE_GUARD   |        20 |         0.6      |          0.000115305 |  5.99518e-05 | True             |
| OpenBMI   | ERM2_REFERENCE    | ALL     |    0    |       1 | TRUE_GUARD   |        15 |         0.6      |          9.32465e-05 | -8.26975e-05 | True             |
| OpenBMI   | ERM2_REFERENCE    | ALL     |    0    |       2 | TRUE_GUARD   |        15 |         0.6      |          9.16183e-05 |  3.23365e-05 | True             |
| OpenBMI   | ERM2_REFERENCE    | ALL     |    0    |       3 | TRUE_GUARD   |        20 |         0.3      |          5.16482e-05 | -7.62574e-05 | True             |
| OpenBMI   | ERM2_REFERENCE    | ALL     |    0    |       4 | TRUE_GUARD   |        15 |         0.4      |          5.48134e-05 | -5.45849e-05 | True             |
| OpenBMI   | ERM2_REFERENCE    | ALL     |    0    |       5 | TRUE_GUARD   |        20 |         0.6      |          0.000113628 |  6.05494e-05 | True             |
| OpenBMI   | P1_ALL_CAP05      | ALL     |    0.05 |       1 | TRUE_GUARD   |        15 |         0.4      |          7.39604e-05 | -0.000125637 | True             |
| OpenBMI   | P1_ALL_CAP05      | ALL     |    0.05 |       2 | TRUE_GUARD   |        15 |         0.6      |          0.000157727 |  0.000105679 | True             |
| OpenBMI   | P1_ALL_CAP05      | ALL     |    0.05 |       3 | TRUE_GUARD   |        20 |         0.45     |          3.81432e-05 | -6.01336e-05 | True             |
| OpenBMI   | P1_ALL_CAP05      | ALL     |    0.05 |       4 | TRUE_GUARD   |        15 |         0.533333 |          0.000117007 |  1.41313e-05 | True             |
| OpenBMI   | P1_ALL_CAP05      | ALL     |    0.05 |       5 | TRUE_GUARD   |        20 |         0.6      |          0.000113304 |  5.8493e-05  | True             |
| OpenBMI   | P2_ALL_CAP10      | ALL     |    0.1  |       1 | TRUE_GUARD   |        15 |         0.4      |          7.39187e-05 | -0.000125404 | True             |
| OpenBMI   | P2_ALL_CAP10      | ALL     |    0.1  |       2 | TRUE_GUARD   |        15 |         0.6      |          0.000143215 |  9.13848e-05 | True             |
| OpenBMI   | P2_ALL_CAP10      | ALL     |    0.1  |       3 | TRUE_GUARD   |        20 |         0.45     |          3.74023e-05 | -6.1265e-05  | True             |
| OpenBMI   | P2_ALL_CAP10      | ALL     |    0.1  |       4 | TRUE_GUARD   |        15 |         0.533333 |          0.000111431 |  1.27047e-05 | True             |
| OpenBMI   | P2_ALL_CAP10      | ALL     |    0.1  |       5 | TRUE_GUARD   |        20 |         0.6      |          0.000113282 |  5.87329e-05 | True             |
| OpenBMI   | P3_ALL_CAP20      | ALL     |    0.2  |       1 | TRUE_GUARD   |        15 |         0.4      |          7.39763e-05 | -0.000125136 | True             |
| OpenBMI   | P3_ALL_CAP20      | ALL     |    0.2  |       2 | TRUE_GUARD   |        15 |         0.6      |          0.000129713 |  7.80086e-05 | True             |
| OpenBMI   | P3_ALL_CAP20      | ALL     |    0.2  |       3 | TRUE_GUARD   |        20 |         0.45     |          3.7358e-05  | -6.12974e-05 | True             |
| OpenBMI   | P3_ALL_CAP20      | ALL     |    0.2  |       4 | TRUE_GUARD   |        15 |         0.533333 |          0.000111576 |  1.39465e-05 | True             |
| OpenBMI   | P3_ALL_CAP20      | ALL     |    0.2  |       5 | TRUE_GUARD   |        20 |         0.6      |          0.000113174 |  5.8639e-05  | True             |
| OpenBMI   | P4_LATE_CAP05     | LATE    |    0.05 |       1 | TRUE_GUARD   |        15 |         0.4      |          8.17766e-05 | -0.000121076 | True             |
| OpenBMI   | P4_LATE_CAP05     | LATE    |    0.05 |       2 | TRUE_GUARD   |        15 |         0.666667 |          0.000168471 |  0.000124672 | True             |
| OpenBMI   | P4_LATE_CAP05     | LATE    |    0.05 |       3 | TRUE_GUARD   |        20 |         0.5      |          4.29921e-05 | -5.5448e-05  | True             |
| OpenBMI   | P4_LATE_CAP05     | LATE    |    0.05 |       4 | TRUE_GUARD   |        15 |         0.533333 |          0.000125111 |  1.89215e-05 | True             |
| OpenBMI   | P4_LATE_CAP05     | LATE    |    0.05 |       5 | TRUE_GUARD   |        20 |         0.6      |          0.000115339 |  5.98542e-05 | True             |
| OpenBMI   | P5_LATE_CAP10     | LATE    |    0.1  |       1 | TRUE_GUARD   |        15 |         0.4      |          8.11299e-05 | -0.000121206 | True             |
| OpenBMI   | P5_LATE_CAP10     | LATE    |    0.1  |       2 | TRUE_GUARD   |        15 |         0.666667 |          0.000167332 |  0.000122888 | True             |
| OpenBMI   | P5_LATE_CAP10     | LATE    |    0.1  |       3 | TRUE_GUARD   |        20 |         0.45     |          4.11607e-05 | -5.74112e-05 | True             |
| OpenBMI   | P5_LATE_CAP10     | LATE    |    0.1  |       4 | TRUE_GUARD   |        15 |         0.533333 |          0.000125548 |  1.93288e-05 | True             |
| OpenBMI   | P5_LATE_CAP10     | LATE    |    0.1  |       5 | TRUE_GUARD   |        20 |         0.6      |          0.000115362 |  5.97186e-05 | True             |
| OpenBMI   | P6_LATE_CAP20     | LATE    |    0.2  |       1 | TRUE_GUARD   |        15 |         0.4      |          7.96924e-05 | -0.000123536 | True             |
| OpenBMI   | P6_LATE_CAP20     | LATE    |    0.2  |       2 | TRUE_GUARD   |        15 |         0.666667 |          0.000164937 |  0.000120289 | True             |
| OpenBMI   | P6_LATE_CAP20     | LATE    |    0.2  |       3 | TRUE_GUARD   |        20 |         0.45     |          3.89293e-05 | -6.00278e-05 | True             |
| OpenBMI   | P6_LATE_CAP20     | LATE    |    0.2  |       4 | TRUE_GUARD   |        15 |         0.533333 |          0.000125615 |  2.00937e-05 | True             |
| OpenBMI   | P6_LATE_CAP20     | LATE    |    0.2  |       5 | TRUE_GUARD   |        20 |         0.6      |          0.000115518 |  5.95659e-05 | True             |
| OpenBMI   | TASK_ONLY_MATCHED | ALL     |    0    |       1 | TRUE_GUARD   |        15 |         0.4      |          8.24342e-05 | -0.000120982 | True             |
| OpenBMI   | TASK_ONLY_MATCHED | ALL     |    0    |       2 | TRUE_GUARD   |        15 |         0.6      |          0.000169882 |  0.000126857 | True             |
| OpenBMI   | TASK_ONLY_MATCHED | ALL     |    0    |       3 | TRUE_GUARD   |        20 |         0.5      |          4.59678e-05 | -5.25244e-05 | True             |
| OpenBMI   | TASK_ONLY_MATCHED | ALL     |    0    |       4 | TRUE_GUARD   |        15 |         0.533333 |          0.000124417 |  1.82569e-05 | True             |
| OpenBMI   | TASK_ONLY_MATCHED | ALL     |    0    |       5 | TRUE_GUARD   |        20 |         0.6      |          0.000115305 |  5.99518e-05 | True             |
| WBCIC     | CAP_ZERO_IDENTITY | ALL     |    0    |       1 | TRUE_GUARD   |        25 |         0.4      |          2.43926e-05 | -3.2801e-05  | True             |
| WBCIC     | CAP_ZERO_IDENTITY | ALL     |    0    |       2 | TRUE_GUARD   |        25 |         0.32     |          1.91391e-05 | -5.55497e-05 | True             |
| WBCIC     | CAP_ZERO_IDENTITY | ALL     |    0    |       3 | TRUE_GUARD   |        25 |         0.6      |          5.15091e-05 |  1.63454e-05 | True             |
| WBCIC     | CAP_ZERO_IDENTITY | ALL     |    0    |       4 | TRUE_GUARD   |        25 |         0.36     |          2.42013e-05 | -3.59505e-05 | True             |
| WBCIC     | CAP_ZERO_IDENTITY | ALL     |    0    |       5 | TRUE_GUARD   |        29 |         0.517241 |          2.96497e-05 | -9.59789e-06 | True             |
| WBCIC     | ERM2_REFERENCE    | ALL     |    0    |       1 | TRUE_GUARD   |        25 |         0.48     |          3.15398e-05 | -2.67565e-05 | True             |
| WBCIC     | ERM2_REFERENCE    | ALL     |    0    |       2 | TRUE_GUARD   |        25 |         0.32     |          1.50657e-05 | -7.58755e-05 | True             |
| WBCIC     | ERM2_REFERENCE    | ALL     |    0    |       3 | TRUE_GUARD   |        25 |         0.44     |          3.5243e-05  | -9.03308e-06 | True             |
| WBCIC     | ERM2_REFERENCE    | ALL     |    0    |       4 | TRUE_GUARD   |        25 |         0.4      |          2.03234e-05 | -3.23689e-05 | True             |
| WBCIC     | ERM2_REFERENCE    | ALL     |    0    |       5 | TRUE_GUARD   |        29 |         0.413793 |          3.65397e-05 | -2.01125e-05 | True             |
| WBCIC     | P1_ALL_CAP05      | ALL     |    0.05 |       1 | TRUE_GUARD   |        25 |         0.4      |          2.40147e-05 | -3.2866e-05  | True             |
| WBCIC     | P1_ALL_CAP05      | ALL     |    0.05 |       2 | TRUE_GUARD   |        25 |         0.32     |          1.82176e-05 | -6.00034e-05 | True             |
| WBCIC     | P1_ALL_CAP05      | ALL     |    0.05 |       3 | TRUE_GUARD   |        25 |         0.6      |          4.96686e-05 |  1.36727e-05 | True             |
| WBCIC     | P1_ALL_CAP05      | ALL     |    0.05 |       4 | TRUE_GUARD   |        25 |         0.36     |          2.41458e-05 | -3.49057e-05 | True             |
| WBCIC     | P1_ALL_CAP05      | ALL     |    0.05 |       5 | TRUE_GUARD   |        29 |         0.482759 |          2.73678e-05 | -1.11928e-05 | True             |
| WBCIC     | P2_ALL_CAP10      | ALL     |    0.1  |       1 | TRUE_GUARD   |        25 |         0.4      |          2.39706e-05 | -3.16536e-05 | True             |
| WBCIC     | P2_ALL_CAP10      | ALL     |    0.1  |       2 | TRUE_GUARD   |        25 |         0.32     |          1.81603e-05 | -6.05047e-05 | True             |
| WBCIC     | P2_ALL_CAP10      | ALL     |    0.1  |       3 | TRUE_GUARD   |        25 |         0.6      |          4.96745e-05 |  1.36238e-05 | True             |
| WBCIC     | P2_ALL_CAP10      | ALL     |    0.1  |       4 | TRUE_GUARD   |        25 |         0.36     |          2.41685e-05 | -3.45951e-05 | True             |
| WBCIC     | P2_ALL_CAP10      | ALL     |    0.1  |       5 | TRUE_GUARD   |        29 |         0.482759 |          2.69315e-05 | -1.11137e-05 | True             |
| WBCIC     | P3_ALL_CAP20      | ALL     |    0.2  |       1 | TRUE_GUARD   |        25 |         0.4      |          2.39599e-05 | -3.04329e-05 | True             |
| WBCIC     | P3_ALL_CAP20      | ALL     |    0.2  |       2 | TRUE_GUARD   |        25 |         0.32     |          1.81639e-05 | -6.05321e-05 | True             |
| WBCIC     | P3_ALL_CAP20      | ALL     |    0.2  |       3 | TRUE_GUARD   |        25 |         0.6      |          4.97133e-05 |  1.36501e-05 | True             |
| WBCIC     | P3_ALL_CAP20      | ALL     |    0.2  |       4 | TRUE_GUARD   |        25 |         0.36     |          2.41387e-05 | -3.45308e-05 | True             |
| WBCIC     | P3_ALL_CAP20      | ALL     |    0.2  |       5 | TRUE_GUARD   |        29 |         0.482759 |          2.68945e-05 | -1.11281e-05 | True             |
| WBCIC     | P4_LATE_CAP05     | LATE    |    0.05 |       1 | TRUE_GUARD   |        25 |         0.4      |          2.39772e-05 | -3.32594e-05 | True             |
| WBCIC     | P4_LATE_CAP05     | LATE    |    0.05 |       2 | TRUE_GUARD   |        25 |         0.32     |          1.83582e-05 | -5.66316e-05 | True             |
| WBCIC     | P4_LATE_CAP05     | LATE    |    0.05 |       3 | TRUE_GUARD   |        25 |         0.56     |          5.0329e-05  |  1.53708e-05 | True             |
| WBCIC     | P4_LATE_CAP05     | LATE    |    0.05 |       4 | TRUE_GUARD   |        25 |         0.36     |          2.39098e-05 | -3.56364e-05 | True             |
| WBCIC     | P4_LATE_CAP05     | LATE    |    0.05 |       5 | TRUE_GUARD   |        29 |         0.517241 |          2.92525e-05 | -9.57785e-06 | True             |
| WBCIC     | P5_LATE_CAP10     | LATE    |    0.1  |       1 | TRUE_GUARD   |        25 |         0.4      |          2.39366e-05 | -3.32278e-05 | True             |
| WBCIC     | P5_LATE_CAP10     | LATE    |    0.1  |       2 | TRUE_GUARD   |        25 |         0.32     |          1.81293e-05 | -5.73051e-05 | True             |
| WBCIC     | P5_LATE_CAP10     | LATE    |    0.1  |       3 | TRUE_GUARD   |        25 |         0.56     |          4.97603e-05 |  1.45239e-05 | True             |
| WBCIC     | P5_LATE_CAP10     | LATE    |    0.1  |       4 | TRUE_GUARD   |        25 |         0.36     |          2.36523e-05 | -3.50463e-05 | True             |
| WBCIC     | P5_LATE_CAP10     | LATE    |    0.1  |       5 | TRUE_GUARD   |        29 |         0.517241 |          2.87258e-05 | -9.71247e-06 | True             |
| WBCIC     | P6_LATE_CAP20     | LATE    |    0.2  |       1 | TRUE_GUARD   |        25 |         0.4      |          2.39116e-05 | -3.34775e-05 | True             |
| WBCIC     | P6_LATE_CAP20     | LATE    |    0.2  |       2 | TRUE_GUARD   |        25 |         0.32     |          1.79255e-05 | -5.77903e-05 | True             |
| WBCIC     | P6_LATE_CAP20     | LATE    |    0.2  |       3 | TRUE_GUARD   |        25 |         0.56     |          4.88412e-05 |  1.32662e-05 | True             |
| WBCIC     | P6_LATE_CAP20     | LATE    |    0.2  |       4 | TRUE_GUARD   |        25 |         0.32     |          2.32697e-05 | -3.49528e-05 | True             |
| WBCIC     | P6_LATE_CAP20     | LATE    |    0.2  |       5 | TRUE_GUARD   |        29 |         0.517241 |          2.75579e-05 | -1.02207e-05 | True             |
| WBCIC     | TASK_ONLY_MATCHED | ALL     |    0    |       1 | TRUE_GUARD   |        25 |         0.4      |          2.43926e-05 | -3.2801e-05  | True             |
| WBCIC     | TASK_ONLY_MATCHED | ALL     |    0    |       2 | TRUE_GUARD   |        25 |         0.32     |          1.91391e-05 | -5.55497e-05 | True             |
| WBCIC     | TASK_ONLY_MATCHED | ALL     |    0    |       3 | TRUE_GUARD   |        25 |         0.6      |          5.15091e-05 |  1.63454e-05 | True             |
| WBCIC     | TASK_ONLY_MATCHED | ALL     |    0    |       4 | TRUE_GUARD   |        25 |         0.36     |          2.42013e-05 | -3.59505e-05 | True             |
| WBCIC     | TASK_ONLY_MATCHED | ALL     |    0    |       5 | TRUE_GUARD   |        29 |         0.517241 |          2.96497e-05 | -9.59789e-06 | True             |
