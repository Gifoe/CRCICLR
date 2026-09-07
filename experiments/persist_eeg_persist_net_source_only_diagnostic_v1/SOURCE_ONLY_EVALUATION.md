# Source-only evaluation

Matched protocol: the same 40 OpenBMI V8_SEARCH subjects, 5 folds × 3 seeds, identical source normalizers, and outcome Session 2 trials. Statistical unit for inference is subject after averaging the three seeds.

| label | BA | macro_f1 | Delta_BA_vs_B0 | Delta_CI95_L | Delta_CI95_U |
| --- | --- | --- | --- | --- | --- |
| Vanilla EEGNet | 0.786167 | 0.782344 | 0.000000 | 0.000000 | 0.000000 |
| Strong EEGNet | 0.791500 | 0.788605 | 0.005333 | -0.002667 | 0.014333 |
| Dual task-only source | 0.777667 | 0.774315 | -0.008500 | -0.020167 | 0.002333 |
| PUD source-only | 0.756500 | 0.753981 | -0.029667 | -0.043083 | -0.016833 |
| Identity source-only | 0.767833 | 0.764648 | -0.018333 | -0.028583 | -0.008165 |
| Random source-only | 0.765333 | 0.762525 | -0.020833 | -0.031752 | -0.010167 |

PUD source-only BA = 0.756500; Macro-F1 = 0.753981. Relative to Vanilla EEGNet, ΔBA = -0.029667, 95% CI [-0.043417, -0.016998]. Relative to the capacity-matched dual source control, ΔBA = -0.021167. Relative to Strong EEGNet, ΔBA = -0.035000.

All source models remained in `eval()`; file hashes and state hashes (parameters plus buffers) were identical before and after inference. Optimizer steps = 0; adaptation calls = 0.
