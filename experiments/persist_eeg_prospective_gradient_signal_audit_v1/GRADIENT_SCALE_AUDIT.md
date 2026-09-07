# Gradient scale audit

The original PMG-fast gradient was evaluated at the frozen M0 anchors for the same 500 paired observations. No parameter update was performed here.

| Dataset | median ||gB_future||/||gA|| | median ||0.5 g_harm||/||gA|| | median ||g_combined||/||gA|| | clip-trigger fraction | active-harm fraction |
|---|---:|---:|---:|---:|---:|
| OpenBMI | 0.979122 | 0.308466 | 1.571506 | 0.7280 | 0.4427 |
| WBCIC | 0.977052 | 0.321820 | 1.565445 | 0.0240 | 0.5084 |

The `0.5*g_harm` term is the exact PMG-fast harm-gradient coefficient. A small scalar ReLU harm does not scale this gradient; when active, its derivative is a task-loss gradient. The per-observation ratios and cosine geometry are in `results/GRADIENT_SCALE_RESULTS.csv`.
