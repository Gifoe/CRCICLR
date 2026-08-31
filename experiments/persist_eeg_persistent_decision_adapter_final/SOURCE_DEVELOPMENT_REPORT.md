# Source development report

The bounded search contained exactly 12 recipes: rank 1/2/4, lambda_X
0.5/1.0, and lambda_P=lambda_precision 1e-3/1e-2. One recipe was selected
jointly by the minimum OpenBMI/WBCIC validation delta: **pda_r1_lx0.5_lp0.01**.
No future resource was opened.

Outcome comparisons at the selected recipe:

* OpenBMI full PDA vs population: ΔBA=-0.0436, 95% CI [-0.0618, -0.0267], n=40
* WBCIC full PDA vs population: ΔBA=-0.0118, 95% CI [-0.0204, -0.0036], n=41
* OpenBMI full vs ordinary adapter: ΔBA=-0.0135, 95% CI [-0.0309, +0.0032], n=40
* WBCIC full vs ordinary adapter: ΔBA=-0.0013, 95% CI [-0.0048, +0.0020], n=41
* OpenBMI correct vs wrong: ΔBA=-0.0014, 95% CI [-0.0153, +0.0114], n=40
* WBCIC correct vs wrong: ΔBA=+0.0018, 95% CI [-0.0051, +0.0096], n=41

The gate failed the positive-delta, CI, ordinary-adapter, correct/wrong and
subject-fraction requirements. Exact terminal: **PERSIST_PDA_SOURCE_NOT_SUPPORTED**.
