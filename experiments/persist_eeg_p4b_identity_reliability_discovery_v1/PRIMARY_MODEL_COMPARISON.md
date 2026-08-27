# Primary Model Comparison

Fixed ridge alpha=1. Primary CV leaves out an entire setting.

| model   |   LOSO_setting_RMSE |   leave_one_run_RMSE |
|:--------|--------------------:|---------------------:|
| M0      |           0.0081420 |            0.0080258 |
| MI      |           0.0079673 |            0.0078181 |
| ME      |           0.0215180 |            0.0067819 |
| MADD    |           0.0215730 |            0.0067949 |
| MINT    |           0.0228639 |            0.0069169 |

- RMSE_MI - RMSE_MINT: -0.01489652; 95% CI [-0.032068833931208574, 0.0005675010482946285].
- RMSE_MADD - RMSE_MINT: -0.00129090; 95% CI [-0.004933686191682303, 0.0009433960303707852].
