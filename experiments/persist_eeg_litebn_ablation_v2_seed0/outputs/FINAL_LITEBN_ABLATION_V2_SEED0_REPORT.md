# LiteBN ablation v2 seed-0: frozen multi-session evaluation

B0_FULL_MATCHED was not trained by explicit user override. The official final LiteBN is a non-matched reference, so these are descriptive contrasts rather than matched-control causal effects.

All five fold checkpoints were averaged within each biological subject before subjects were equally averaged. WS-BA is the within-subject minimum session BA after fold averaging.

| Variant | Params range | MACs range | MI BA/WS | ERP BA/WS | SSVEP BA/WS | WBCIC BA/WS | Mean BA | Mean WS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OFFICIAL_FINAL_LITEBN_REFERENCE | 47786-48108 | 99644896-106397024 | 0.7449/0.7113 | 0.8465/0.8150 | 0.9070/0.8979 | 0.8093/0.7352 | 0.8269 | 0.7898 |
| B1_SAME_SCALE_63 | 47658-47980 | 92220896-98461024 | 0.7174/0.6767 | 0.8497/0.8189 | 0.9054/0.8950 | 0.8049/0.7191 | 0.8194 | 0.7774 |
| B2_SCALE_COLLAPSE | 47786-48108 | 99644896-106397024 | 0.6921/0.6616 | 0.8531/0.8208 | 0.8919/0.8856 | 0.7965/0.7131 | 0.8084 | 0.7703 |
| B3_SINGLE_SPATIAL_BASIS | 46778-47004 | 98636896-105293024 | 0.7060/0.6789 | 0.8502/0.8193 | 0.9074/0.9024 | 0.8025/0.7218 | 0.8165 | 0.7806 |
| B4_ONE_STAGE_BACKEND | 41578-41900 | 98884896-105637024 | 0.7377/0.6986 | 0.8381/0.8078 | 0.8731/0.8510 | 0.7890/0.7123 | 0.8095 | 0.7674 |

## Paired effects versus non-matched official reference

| Variant | Task | Delta future BA [95% CI], pp | Delta WS-BA [95% CI], pp |
|---|---|---:|---:|
| B1_SAME_SCALE_63 | OpenBMI_MI | -2.743 [-5.443, +0.071] | -3.457 [-5.614, -1.143] |
| B2_SCALE_COLLAPSE | OpenBMI_MI | -5.271 [-7.271, -3.186] | -4.971 [-7.014, -2.971] |
| B3_SINGLE_SPATIAL_BASIS | OpenBMI_MI | -3.886 [-6.114, -1.286] | -3.243 [-5.614, -0.585] |
| B4_ONE_STAGE_BACKEND | OpenBMI_MI | -0.714 [-2.529, +1.129] | -1.271 [-3.014, +0.543] |
| B1_SAME_SCALE_63 | OpenBMI_ERP | +0.320 [-0.035, +0.684] | +0.383 [-0.013, +0.771] |
| B2_SCALE_COLLAPSE | OpenBMI_ERP | +0.658 [+0.184, +1.171] | +0.580 [+0.123, +1.085] |
| B3_SINGLE_SPATIAL_BASIS | OpenBMI_ERP | +0.364 [-0.114, +0.900] | +0.430 [-0.267, +1.078] |
| B4_ONE_STAGE_BACKEND | OpenBMI_ERP | -0.844 [-1.136, -0.576] | -0.725 [-1.106, -0.393] |
| B1_SAME_SCALE_63 | OpenBMI_SSVEP | -0.157 [-1.200, +0.957] | -0.286 [-1.300, +0.743] |
| B2_SCALE_COLLAPSE | OpenBMI_SSVEP | -1.514 [-3.014, -0.300] | -1.229 [-2.329, -0.229] |
| B3_SINGLE_SPATIAL_BASIS | OpenBMI_SSVEP | +0.043 [-1.086, +1.214] | +0.457 [-0.500, +1.457] |
| B4_ONE_STAGE_BACKEND | OpenBMI_SSVEP | -3.386 [-5.500, -1.486] | -4.686 [-7.629, -2.229] |
| B1_SAME_SCALE_63 | WBCIC_MI | -0.440 [-1.630, +0.710] | -1.610 [-3.200, -0.330] |
| B2_SCALE_COLLAPSE | WBCIC_MI | -1.280 [-2.510, -0.100] | -2.210 [-4.020, -0.570] |
| B3_SINGLE_SPATIAL_BASIS | WBCIC_MI | -0.680 [-2.180, +0.830] | -1.340 [-3.010, +0.110] |
| B4_ONE_STAGE_BACKEND | WBCIC_MI | -2.030 [-3.650, -0.520] | -2.290 [-3.540, -0.890] |
