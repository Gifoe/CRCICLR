# BNLOCK Stage-2 Phase-A Decision

| Metric | OpenBMI | WBCIC |
|---|---:|---:|
| EEGNet BA | 0.7577 | 0.7827 |
| LiteBN BA | 0.7915 | 0.7870 |
| Frozen LOGIT50 BA | 0.8000 | 0.7940 |
| Old JOINT-CE BA | 0.7995 | 0.7463 |
| BNLOCK-JOINTCE BA | 0.8005 | 0.7951 |
| BNLOCK-JOINT delta vs EEGNet (pp) | 4.2750 | 1.2429 |
| BNLOCK-JOINT delta vs LOGIT50 (pp) | 0.0500 | 0.1126 |
| Positive folds | 5.0000 | 4.0000 |
| Harm <= -1pp | 0.1250 | 0.2581 |

BNLOCK-NRRF BA: OpenBMI 0.8005; WBCIC 0.7941.
BNLOCK-NRRF delta vs EEGNet: OpenBMI +4.275 pp; WBCIC +1.146 pp.
BN-lock fixed previous collapse: YES.
JOINT adds value beyond frozen fusion: NO.
NRRF adds value beyond JOINT: NO.
Exact gate: {"added_value": false, "bnlock_repairs_previous_collapse": true, "frozen_wbcic_harm_le_minus_1pp": 0.16129032258064516, "nrrf_adds_value": false, "openbmi_primary": false, "openbmi_upside_preserved": true, "preferred_method": null, "primary_pass": false, "terminal": "BNLOCK_REPAIRS_COLLAPSE_BUT_NO_ADDED_VALUE_STOP", "wbcic_primary": false}.
Final terminal: **BNLOCK_REPAIRS_COLLAPSE_BUT_NO_ADDED_VALUE_STOP**.
