# V2 action library selection

| dataset   | action                  |   alpha |    mean_gain |   win_rate |   harm_rate |   safe_beneficial_rate |   unavailable_rate |
|:----------|:------------------------|--------:|-------------:|-----------:|------------:|-----------------------:|-------------------:|
| eegmmidb  | no_tta                  |     0.1 |  0           |  0         |   0         |              0         |          0         |
| eegmmidb  | no_tta                  |     0.2 |  0           |  0         |   0         |              0         |          0         |
| eegmmidb  | official_t3a            |     0.1 | -0.00147222  |  0.353333  |   0.42      |              0.346667  |          0         |
| eegmmidb  | official_t3a            |     0.2 | -0.00147222  |  0.353333  |   0.42      |              0.353333  |          0         |
| eegmmidb  | robust_residual_adapter |     0.1 | -0.000393275 |  0.0733333 |   0.0933333 |              0.0733333 |          0.02      |
| eegmmidb  | robust_residual_adapter |     0.2 | -0.000393275 |  0.0733333 |   0.0933333 |              0.0733333 |          0.02      |
| hmc       | no_tta                  |     0.1 |  0           |  0         |   0         |              0         |          0         |
| hmc       | no_tta                  |     0.2 |  0           |  0         |   0         |              0         |          0         |
| hmc       | official_t3a            |     0.1 | -0.00702381  |  0.417143  |   0.548571  |              0.365714  |          0         |
| hmc       | official_t3a            |     0.2 | -0.00702381  |  0.417143  |   0.548571  |              0.394286  |          0         |
| hmc       | robust_residual_adapter |     0.1 | -0.000452381 |  0.131429  |   0.217143  |              0.131429  |          0.0228571 |
| hmc       | robust_residual_adapter |     0.2 | -0.000452381 |  0.131429  |   0.217143  |              0.131429  |          0.0228571 |

| dataset   |   seed |   alpha |   safe_oracle_gain |   ci_lower |   ci_upper |   safe_beneficial_subject_rate |
|:----------|-------:|--------:|-------------------:|-----------:|-----------:|-------------------------------:|
| hmc       |      0 |     0.1 |         0.0246429  | 0.0113095  |  0.0422649 |                       0.485714 |
| hmc       |      0 |     0.2 |         0.0271429  | 0.0130923  |  0.0454792 |                       0.542857 |
| hmc       |      1 |     0.1 |         0.0225     | 0.0115446  |  0.0346429 |                       0.514286 |
| hmc       |      1 |     0.2 |         0.0225     | 0.0117857  |  0.0352411 |                       0.514286 |
| hmc       |      2 |     0.1 |         0.0169048  | 0.00857143 |  0.0265476 |                       0.571429 |
| hmc       |      2 |     0.2 |         0.0169048  | 0.00857143 |  0.0265476 |                       0.571429 |
| hmc       |      3 |     0.1 |         0.014881   | 0.00547321 |  0.0261905 |                       0.428571 |
| hmc       |      3 |     0.2 |         0.0153571  | 0.00607143 |  0.027619  |                       0.457143 |
| hmc       |      4 |     0.1 |         0.00583333 | 0.0022619  |  0.01      |                       0.342857 |
| hmc       |      4 |     0.2 |         0.00666667 | 0.00297619 |  0.0109524 |                       0.4      |
| eegmmidb  |      0 |     0.1 |         0.0119444  | 0.00611111 |  0.0194444 |                       0.4      |
| eegmmidb  |      0 |     0.2 |         0.0125     | 0.00638889 |  0.0202778 |                       0.433333 |
| eegmmidb  |      1 |     0.1 |         0.0127778  | 0.00722222 |  0.0188889 |                       0.433333 |
| eegmmidb  |      1 |     0.2 |         0.0127778  | 0.00720833 |  0.0194444 |                       0.433333 |
| eegmmidb  |      2 |     0.1 |         0.00722222 | 0.00333333 |  0.0122222 |                       0.3      |
| eegmmidb  |      2 |     0.2 |         0.00722222 | 0.00333333 |  0.0122222 |                       0.3      |
| eegmmidb  |      3 |     0.1 |         0.0111111  | 0.005      |  0.0183333 |                       0.333333 |
| eegmmidb  |      3 |     0.2 |         0.0111111  | 0.005      |  0.0183472 |                       0.333333 |
| eegmmidb  |      4 |     0.1 |         0.0168348  | 0.00891813 |  0.02625   |                       0.433333 |
| eegmmidb  |      4 |     0.2 |         0.0168348  | 0.00839108 |  0.0268668 |                       0.433333 |

Gate: **PASS**. Formal library is `no_tta`, `official_t3a`, and `robust_residual_adapter`.
Tent and official EATA configure BatchNorm statistics/affine parameters; the frozen CBraMod and selected heads contain LayerNorm and no eligible BatchNorm, so they are marked incompatible rather than relabeled. Official SAR was not vendored or claimed: a custom LayerNorm/SAM variant would be a different method and is outside the frozen two-TTA library.
