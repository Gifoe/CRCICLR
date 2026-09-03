# Random guard audit

The selected PSG candidate was rerun with a deterministic B batch from a different pseudo-future subject group. The true/random BA and held-out probe comparison is in `results/RANDOM_GUARD_CONTROL.csv`.

| dataset   | candidate     | scope   |   kappa |   epoch |   true_guard_BA |   random_guard_BA |   true_guard_delta_pp |   random_guard_delta_pp |   true_vs_random_guard_pp |   true_probe_harm_frequency |   random_probe_harm_frequency |   true_probe_mean_positive_harm |   random_probe_mean_positive_harm |
|:----------|:--------------|:--------|--------:|--------:|----------------:|------------------:|----------------------:|------------------------:|--------------------------:|----------------------------:|------------------------------:|--------------------------------:|----------------------------------:|
| OpenBMI   | P6_LATE_CAP20 | LATE    |     0.2 |       2 |        0.81963  |          0.82     |             0.0555556 |               0.0925926 |                -0.037037  |                    0.666667 |                      0.333333 |                     0.000164937 |                       4.76489e-05 |
| WBCIC     | P6_LATE_CAP20 | LATE    |     0.2 |       2 |        0.790318 |          0.790074 |             0.401931  |               0.377541  |                 0.0243902 |                    0.32     |                      0.4      |                     1.79255e-05 |                       3.9767e-05  |
