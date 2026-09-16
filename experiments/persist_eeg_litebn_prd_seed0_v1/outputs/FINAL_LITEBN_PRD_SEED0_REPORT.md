# LiteBN CE versus CE+PRD seed-0

PRD lambda=0.25 was reused without tuning.

| Task | Condition | Alignment | Future BA | WS-BA |
|---|---|---:|---:|---:|
| OpenBMI_MI | CE | 0.7776 | 0.7064 | 0.6691 |
| OpenBMI_MI | CE_PLUS_PRD | 0.8782 | 0.6901 | 0.6593 |
| WBCIC_MI | CE | 0.7789 | 0.7461 | 0.6775 |
| WBCIC_MI | CE_PLUS_PRD | 0.8078 | 0.7218 | 0.6547 |

| Task | Delta alignment | Delta future BA pp [95% CI] | Delta WS-BA pp [95% CI] |
|---|---:|---:|---:|
| OpenBMI_MI | +0.1006 [+0.0633, +0.1379] | -1.629 [-2.529, -0.771] | -0.986 [-1.829, -0.143] |
| WBCIC_MI | +0.0289 [-0.0339, +0.0791] | -2.430 [-3.590, -1.310] | -2.280 [-3.600, -1.090] |
