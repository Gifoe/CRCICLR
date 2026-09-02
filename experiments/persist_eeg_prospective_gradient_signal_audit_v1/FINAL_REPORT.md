# Prospective Gradient Signal Audit v1

terminal = PROSPECTIVE_GRADIENT_SIGNAL_SUPPORTED

## Signal test (gradient-pair biological environments)

| Dataset | Spearman rho | 95% cluster CI | sign accuracy | conflict rate | harm AUROC | conflict minus nonconflict actual Delta_B | 95% CI | signal pass |
|---|---:|---|---:|---:|---:|---:|---|---|
| OpenBMI | 0.999131 | [0.998022, 0.999250] | 0.9880 | 0.3400 | 0.998948 | 0.0026695746 | [0.0022750941, 0.0030362722] | YES |
| WBCIC | 0.999849 | [0.999673, 0.999849] | 1.0000 | 0.5200 | 1.000000 | 0.00068801856 | [0.00061710723, 0.00076827533] | YES |

## Gradient scale

| Dataset | median ||gB||/||gA|| | median ||0.5gH||/||gA|| | median ||gCombined||/||gA|| | clip-trigger fraction | active-harm fraction |
|---|---:|---:|---:|---:|---:|
| OpenBMI | 0.979122 | 0.308466 | 1.571506 | 0.7280 | 0.4427 |
| WBCIC | 0.977052 | 0.321820 | 1.565445 | 0.0240 | 0.5084 |

## WBCIC fold-1 collapse diagnosis

`D_batchnorm_running_stat_drift`. Details are in `WBCIC_FOLD1_FORENSIC.md`; the forensic reproduction used the exact original five-epoch PMG-fast recipe once and did not repair it.

## Required answers

1. Source gradient conflict predicts actual pseudo-future harm? Yes under the predeclared gates.
2. Reproducible in both datasets? Yes.
3. Is PMG-fast's harm gradient disproportionate to its tiny scalar harm? See the scale table and per-observation audit; the derivative is not magnitude-weighted by the tiny scalar.
4. Most likely WBCIC fold-1 cause: D_batchnorm_running_stat_drift.
5. Direct prospective-gradient safeguarding scientifically justified? Only as a follow-up research question; this audit does not validate a method.
6. PMG / prospective-gradient family: literature audit before any design, per protocol.
7. Outcome/sealed subjects accessed? No.
8. Seed 1/2 run? No.

Runtime seconds: 90.564; primary observations: 500; source-only: yes; mathematical audit: PASS.
