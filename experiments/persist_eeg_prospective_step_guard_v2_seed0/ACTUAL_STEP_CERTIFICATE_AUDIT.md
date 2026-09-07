# Actual-step certificate audit

For TASK_ONLY_MATCHED and each registered candidate, h = g_B^T Delta_task was compared with the measured guard-batch loss change. The machine-readable aggregate is `results/ACTUAL_STEP_CERTIFICATE.csv`; selection never used this table.

| dataset   | candidate         | scope   |   kappa | guard_kind   |   n_steps |   spearman_rho |   pearson_r |   sign_accuracy |   harm_auroc |   mean_h_before |   mean_delta_L_B_guard |
|:----------|:------------------|:--------|--------:|:-------------|----------:|---------------:|------------:|----------------:|-------------:|----------------:|-----------------------:|
| OpenBMI   | CAP_ZERO_IDENTITY | ALL     |    0    | TRUE_GUARD   |      1705 |       0.999908 |    0.999912 |        0.995894 |     0.999946 |    -3.08993e-05 |           -2.83114e-05 |
| OpenBMI   | ERM2_REFERENCE    | ALL     |    0    | TRUE_GUARD   |      1705 |       0.999957 |    0.999963 |        0.999413 |     0.999967 |    -0.00047409  |           -0.000470956 |
| OpenBMI   | P1_ALL_CAP05      | ALL     |    0.05 | TRUE_GUARD   |      1705 |       0.9616   |    0.954222 |        0.995894 |     0.998204 |    -2.93474e-05 |           -8.42611e-05 |
| OpenBMI   | P2_ALL_CAP10      | ALL     |    0.1  | TRUE_GUARD   |      1705 |       0.917852 |    0.884125 |        0.995308 |     0.997621 |    -2.9038e-05  |           -9.78313e-05 |
| OpenBMI   | P3_ALL_CAP20      | ALL     |    0.2  | TRUE_GUARD   |      1705 |       0.909665 |    0.858391 |        0.994721 |     0.99693  |    -2.89932e-05 |           -9.98938e-05 |
| OpenBMI   | P4_LATE_CAP05     | LATE    |    0.05 | TRUE_GUARD   |      1705 |       0.997899 |    0.997388 |        0.994721 |     0.999628 |    -3.07073e-05 |           -4.72831e-05 |
| OpenBMI   | P5_LATE_CAP10     | LATE    |    0.1  | TRUE_GUARD   |      1705 |       0.991752 |    0.990006 |        0.995894 |     0.999624 |    -3.05493e-05 |           -6.18122e-05 |
| OpenBMI   | P6_LATE_CAP20     | LATE    |    0.2  | TRUE_GUARD   |      1705 |       0.971278 |    0.967501 |        0.993548 |     0.998412 |    -3.0337e-05  |           -8.05556e-05 |
| OpenBMI   | TASK_ONLY_MATCHED | ALL     |    0    | TRUE_GUARD   |      1705 |       0.999908 |    0.999912 |        0.995894 |     0.999946 |    -3.08993e-05 |           -2.83114e-05 |
| WBCIC     | CAP_ZERO_IDENTITY | ALL     |    0    | TRUE_GUARD   |      2580 |       0.999973 |    0.999913 |        1        |     1        |    -3.18723e-05 |           -3.1116e-05  |
| WBCIC     | ERM2_REFERENCE    | ALL     |    0    | TRUE_GUARD   |      2580 |       0.999968 |    0.999951 |        0.999612 |     1        |    -0.00030127  |           -0.000300212 |
| WBCIC     | P1_ALL_CAP05      | ALL     |    0.05 | TRUE_GUARD   |      2580 |       0.973217 |    0.962613 |        0.993023 |     0.997015 |    -3.17146e-05 |           -5.58519e-05 |
| WBCIC     | P2_ALL_CAP10      | ALL     |    0.1  | TRUE_GUARD   |      2580 |       0.94941  |    0.924937 |        0.992248 |     0.993174 |    -3.17198e-05 |           -6.0852e-05  |
| WBCIC     | P3_ALL_CAP20      | ALL     |    0.2  | TRUE_GUARD   |      2580 |       0.946425 |    0.909438 |        0.992248 |     0.994413 |    -3.17371e-05 |           -6.17282e-05 |
| WBCIC     | P4_LATE_CAP05     | LATE    |    0.05 | TRUE_GUARD   |      2580 |       0.994562 |    0.993637 |        0.994574 |     0.998589 |    -3.1803e-05  |           -4.45368e-05 |
| WBCIC     | P5_LATE_CAP10     | LATE    |    0.1  | TRUE_GUARD   |      2580 |       0.982028 |    0.976142 |        0.994961 |     0.998312 |    -3.1761e-05  |           -5.26968e-05 |
| WBCIC     | P6_LATE_CAP20     | LATE    |    0.2  | TRUE_GUARD   |      2580 |       0.956918 |    0.937952 |        0.993023 |     0.995315 |    -3.17475e-05 |           -5.96246e-05 |
| WBCIC     | TASK_ONLY_MATCHED | ALL     |    0    | TRUE_GUARD   |      2580 |       0.999973 |    0.999913 |        1        |     1        |    -3.18723e-05 |           -3.1116e-05  |
