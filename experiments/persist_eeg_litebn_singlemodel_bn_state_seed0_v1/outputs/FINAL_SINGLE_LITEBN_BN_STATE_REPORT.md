# Single-LiteBN BN-state seed-0 intervention

All adapted checkpoints were frozen on source-only data before heldout evaluation.

| Task | Condition | Future BA | WS-BA |
|---|---|---:|---:|
| OpenBMI_MI | M00_ORIGINAL | 0.7449 | 0.7113 |
| OpenBMI_MI | M11_ADAPTED | 0.7446 | 0.7110 |
| OpenBMI_MI | M10_RESTORE_STATE | 0.7434 | 0.7084 |
| OpenBMI_MI | M01_TRANSPLANT_STATE | 0.7457 | 0.7104 |
| WBCIC_MI | M00_ORIGINAL | 0.8093 | 0.7352 |
| WBCIC_MI | M11_ADAPTED | 0.8078 | 0.7359 |
| WBCIC_MI | M10_RESTORE_STATE | 0.8081 | 0.7343 |
| WBCIC_MI | M01_TRANSPLANT_STATE | 0.8104 | 0.7389 |

| Task | Contrast | Delta future BA pp [95% CI] | Delta WS-BA pp [95% CI] |
|---|---|---:|---:|
| OpenBMI_MI | adaptation_loss | -0.029 [-0.614, +0.586] | -0.029 [-0.829, +0.771] |
| OpenBMI_MI | state_restoration | -0.114 [-0.529, +0.271] | -0.257 [-0.643, +0.143] |
| OpenBMI_MI | state_transplant | +0.086 [-0.314, +0.500] | -0.086 [-0.586, +0.414] |
| OpenBMI_MI | parameter_only_retained | -0.143 [-0.714, +0.400] | -0.286 [-1.029, +0.414] |
| WBCIC_MI | adaptation_loss | -0.150 [-0.540, +0.230] | +0.070 [-1.080, +1.100] |
| WBCIC_MI | state_restoration | +0.030 [-0.390, +0.440] | -0.160 [-0.840, +0.800] |
| WBCIC_MI | state_transplant | +0.110 [-0.320, +0.510] | +0.370 [-0.550, +1.020] |
| WBCIC_MI | parameter_only_retained | -0.120 [-0.640, +0.420] | -0.090 [-0.790, +0.620] |
