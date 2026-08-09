# V3 Oracle headroom report

Stage-0 gate: **GO**. Although the CI lower bounds are positive, relative reductions are below 0.5%, so this is weak headroom.

| dataset   |   alpha |   positive_subject_rate |   relative_set_size_reduction |   relative_ci_lower |   relative_ci_upper |   maximum_single_action_positive_rate |   harm_rate |
|:----------|--------:|------------------------:|------------------------------:|--------------------:|--------------------:|--------------------------------------:|------------:|
| eegmmidb  |  0.1000 |                  0.2378 |                        0.0018 |              0.0012 |              0.0023 |                                0.2400 |      0.0533 |
| eegmmidb  |  0.2000 |                  0.2603 |                        0.0023 |              0.0016 |              0.0031 |                                0.2600 |      0.0533 |
| hmc       |  0.1000 |                  0.4417 |                        0.0036 |              0.0022 |              0.0053 |                                0.4457 |      0.0229 |
| hmc       |  0.2000 |                  0.3470 |                        0.0040 |              0.0024 |              0.0059 |                                0.3600 |      0.0229 |
