# Matched homogeneous-ensemble control decision

Terminal: `HETEROGENEOUS_ADVANTAGE_WEAK`.

| Block | EEG homo G_best | LiteBN homo G_best | Hetero G_best | D_avg | D_best | D_avg 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | -1.171 | -1.414 | -1.045 | +0.248 | -0.964 | [-0.133, +0.664] |
| OpenBMI_ERP | -0.092 | -0.083 | +0.145 | +0.232 | -0.178 | [+0.092, +0.377] |
| OpenBMI_SSVEP | +0.214 | -0.086 | +0.010 | -0.055 | -0.857 | [-0.702, +0.419] |
| WBCIC_MI | -0.553 | -0.723 | -0.688 | -0.050 | -0.875 | [-0.435, +0.217] |

Equal-block global `D_avg`: **+0.094 pp** (95% CI [-0.115, +0.278]).
Equal-block global `D_best`: **-0.719 pp** (95% CI [-0.962, -0.512]).

All five folds, all three unordered seed pairs, and both heterogeneous orientations were retained. No model was trained in this control.
The correct interpretation is limited to whether the frozen heterogeneous 50/50 fusion has added value beyond these matched ordinary two-instance ensemble controls; it does not establish distinct neural mechanisms.
