# V3 action search report

64 T3A and 81 adapter configurations were screened by subject-grouped successive halving. T3A produced no Safe-Oracle contribution; all positive headroom came from the residual adapter.

| dataset   |   seed | action                  | config_id                            |   availability_rate |   mean_safe_gain |   harm_rate |
|:----------|-------:|:------------------------|:-------------------------------------|--------------------:|-----------------:|------------:|
| eegmmidb  |      3 | robust_residual_adapter | robust_residual_adapter-346f250d78e3 |              0.9333 |           0.0097 |      0.1333 |
| eegmmidb  |      1 | robust_residual_adapter | robust_residual_adapter-808e998f02ef |              0.8667 |           0.0042 |      0.2333 |
| eegmmidb  |      4 | robust_residual_adapter | robust_residual_adapter-0903d909f1e8 |              0.8667 |           0.0019 |      0.0333 |
| hmc       |      4 | robust_residual_adapter | robust_residual_adapter-2c51a511fc83 |              1.0000 |           0.0004 |      0.0000 |
| eegmmidb  |      2 | robust_residual_adapter | robust_residual_adapter-211892d39d6a |              0.8667 |           0.0003 |      0.0333 |
| eegmmidb  |      2 | official_t3a            | official_t3a-0e621b80ffab            |              0.0000 |           0.0000 |      0.0000 |
| eegmmidb  |      1 | official_t3a            | official_t3a-0ad43336d54d            |              0.0000 |           0.0000 |      0.0000 |
| hmc       |      1 | official_t3a            | official_t3a-06d8b003eb93            |              0.0000 |           0.0000 |      0.0000 |
| hmc       |      0 | official_t3a            | official_t3a-13d4c0513513            |              0.0000 |           0.0000 |      0.0000 |
| eegmmidb  |      3 | official_t3a            | official_t3a-06d8b003eb93            |              0.0000 |           0.0000 |      0.0000 |
| eegmmidb  |      0 | official_t3a            | official_t3a-06d8b003eb93            |              0.0000 |           0.0000 |      0.0000 |
| hmc       |      2 | official_t3a            | official_t3a-06d8b003eb93            |              0.0000 |           0.0000 |      0.0000 |
| hmc       |      4 | official_t3a            | official_t3a-06d8b003eb93            |              0.0000 |           0.0000 |      0.0000 |
| hmc       |      3 | official_t3a            | official_t3a-13d4c0513513            |              0.0000 |           0.0000 |      0.0000 |
| eegmmidb  |      4 | official_t3a            | official_t3a-0ad43336d54d            |              0.0000 |           0.0000 |      0.0000 |
| hmc       |      3 | robust_residual_adapter | robust_residual_adapter-808e998f02ef |              1.0000 |          -0.0010 |      0.0286 |
| hmc       |      0 | robust_residual_adapter | robust_residual_adapter-346f250d78e3 |              1.0000 |          -0.0018 |      0.0000 |
| hmc       |      2 | robust_residual_adapter | robust_residual_adapter-0fc0ca7c17b4 |              1.0000 |          -0.0027 |      0.0571 |
| eegmmidb  |      0 | robust_residual_adapter | robust_residual_adapter-6e54d31acf02 |              0.9333 |          -0.0031 |      0.1000 |
| hmc       |      1 | robust_residual_adapter | robust_residual_adapter-81635b9b8859 |              0.9429 |          -0.0035 |      0.1429 |
