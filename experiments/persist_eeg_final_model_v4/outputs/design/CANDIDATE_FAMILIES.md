# Candidate families

| Family | First test | Reason | Failure signal |
| --- | --- | --- | --- |
| Calibrated linear | KEEP, KEEP+ACTION, +PERSIST | strongest low-variance stacking control | no grouped gain |
| Shallow HGB | same ladder | limited nonlinear interactions | fold instability/overfit |
| Anchored residual | bounded correction around B6 | reduces hard-switch harm | correction collapses to zero or harms |
| Joint correctness/ranking | score all expert tokens jointly | attacks selection target directly | poor rescue precision |
| DeepSets/MoE | shared token encoder, B6 prior | variable expert count, soft aggregation | no transfer or unstable weights |
| Frozen representation gate | optional only after meta plateau | tests missing representation information | capacity-only gain |

The action grid is finite: AMPLIFY, GEOMETRY, ERASE and alpha 0.25/0.5
interpolations. ERASE is high risk and is never forced.
