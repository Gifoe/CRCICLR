# IID versus matched episodic comparison

The IID control is read from `/root/p4_shin2017b_seed0/seed0_fullmodel_subject_metrics.csv` (SHA-256 `c3b9cce1ac734fa53255565d0e20eedab747bf2646fd9a13778b8301645556a0`); it has not been overwritten.

| Model | IID future BA | Episodic future BA | Change | IID Macro-F1 | Episodic Macro-F1 | Change | IID WS-BA | Episodic WS-BA | Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EEGNet | 0.721 | 0.719 | -0.002 | 0.712 | 0.707 | -0.005 | 0.621 | 0.621 | -0.000 |
| SIRE-EEG | 0.695 | 0.700 | +0.005 | 0.685 | 0.688 | +0.003 | 0.583 | 0.607 | +0.024 |

## Episodic paired SIRE-EEG minus EEGNet

| Metric | Mean difference | 95% bootstrap CI | Biological subjects | Bootstrap resamples |
|---|---:|---:|---:|---:|
| future BA | -0.019 | [-0.067, +0.028] | 29 | 20,000 |
| WS-BA | -0.014 | [-0.062, +0.036] | 29 | 20,000 |

No diagnostics, ScaleCollapse, additional baseline, or seed-1/2 run is included in this experiment.
