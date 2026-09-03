# Mechanism audit

`results/GRADIENT_CONFLICT_LOG.csv` contains every optimizer step. It records conflict rate, joint activation, cosine and correction size. `results/PROSPECTIVE_HARM_LOG.csv` measures the fixed pseudo-future diagnostic B loss every 20 steps and is never used in updates or selection.

| candidate      |    lr |   conflict_rate |   joint_activation_rate |   mean_relative_correction |   median_relative_correction |   mean_cos_gA_gB |
|:---------------|------:|----------------:|------------------------:|---------------------------:|-----------------------------:|-----------------:|
| C0_ERM2        | 3e-05 |        0.499417 |             0           |                0           |                     0        |      0.00147957  |
| C1_SCOPE_A     | 3e-05 |        0.491482 |             0.491482    |                0.925851    |                     0.99558  |      0.0058034   |
| C1_SCOPE_B     | 3e-05 |        0.497316 |             0.497316    |                0.944302    |                     0.998626 |      0.00275575  |
| C2_SCOPE_A     | 3e-05 |        0.509452 |             0.00280047  |                0.000228278 |                     0        |      0.00140187  |
| C2_SCOPE_B     | 3e-05 |        0.509452 |             0.000700117 |                3.27992e-05 |                     0        |     -0.00146662  |
| C3_SCOPE_A     | 3e-05 |        0.505951 |             0.000700117 |                1.92308e-05 |                     0        |      0.00154014  |
| C3_SCOPE_B     | 3e-05 |        0.501284 |             0.000466744 |                1.42775e-05 |                     0        |      0.000514015 |
| C4_LATE_STRICT | 3e-05 |        0.493349 |             0.000466744 |                8.84703e-06 |                     0        |      0.0038591   |
| C5_LATE_SOFT   | 3e-05 |        0.485764 |             0.000233372 |                3.1194e-06  |                     0        |      0.00781316  |
