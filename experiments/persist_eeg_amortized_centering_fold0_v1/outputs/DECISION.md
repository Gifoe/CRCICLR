# Amortized centering fold0 decision

OpenBMI fold0 / seed0

| Model | BA | Macro-F1 |
|---|---:|---:|
| Locked EEGNet-ERM | 0.7964 | 0.7937 |
| Decomp-CE | 0.7229 | 0.7189 |
| Decomp-Offset | 0.7150 | 0.7124 |

Architecture delta: -7.36 pp. Offset-supervision delta: -0.79 pp. Total delta: -8.14 pp.

Mean / median subject total delta: -8.14 / -8.00 pp. Improved/tied/harmed: 0 / 0 / 14. Bootstrap 95% CI: [-10.50, -5.71] pp.

Offset estimate beats zero MSE: YES (MSE 0.048535 versus zero 0.374190; cosine 0.9443). h_rel suppression ratio: 0.9450; lower than one: YES.

Full / rel-only / res-only BA: 0.7150 / 0.6964 / 0.6914. OracleCenter-Diagnostic BA: 0.7286 (TRANSDUCTIVE DIAGNOSTIC ONLY).

Legal method is single-trial at inference: YES. Target-subject history entered legal method: NO. Reserved holdout accessed: NO.

Did variance reduction translate into utility: NO. Supports the stated constructive claim: NO.

Terminal: `DECOMP_ARCHITECTURE_HARM_STOP`.

Recommended next action: do not tune lambda or add offset losses; retain centering as a diagnostic and reconsider the constructive representation mechanism.
