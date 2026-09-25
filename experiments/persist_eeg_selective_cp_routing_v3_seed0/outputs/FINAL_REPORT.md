# Utility-Guided Selective Complement Routing (UGCR-V3)

**Status: POST_HELDOUT_DEVELOPMENT_EVALUATION.** The formal heldout cohort was accessed in prior V1/V2 work; this is not untouched confirmatory evidence. V3 architecture, utility signs, coefficients, shuffle/random controls, metrics, and evaluator were frozen in `V3_PROTOCOL_LOCK.json` before any V3 heldout EEG array was read.

EEGNet was the canonical V1/V2 final-refit checkpoint, fully frozen. No router, gate, adapter, classifier, backbone, or BN state was trained; V3 adds zero trainable parameters. Direction utility was re-estimated only on the non-final-heldout refit-training pool with the frozen V2.5 native-coordinate P-mediated margin estimand.

## Primary future-session subject-equal BA

| Task | Baseline | Positive-only | Shuffled | Random mean | UGCR-V3 | Δ vs Base (95% CI) | Δ vs Shuffled (95% CI) | Δ vs Random mean (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | 0.7114 | 0.7079 | 0.7086 | 0.7121 | 0.7079 | -0.0036 [-0.0121, +0.0057] | -0.0007 [-0.0079, +0.0079] | -0.0043 [-0.0143, +0.0057] |
| OpenBMI_ERP | 0.8645 | 0.8624 | 0.8617 | 0.8645 | 0.8624 | -0.0021 [-0.0048, +0.0002] | +0.0008 [-0.0000, +0.0016] | -0.0020 [-0.0048, +0.0003] |
| OpenBMI_SSVEP | 0.9200 | 0.9243 | 0.9207 | 0.9200 | 0.9243 | +0.0043 [-0.0007, +0.0107] | +0.0036 [-0.0014, +0.0100] | +0.0043 [-0.0007, +0.0107] |
| WBCIC_MI | 0.7905 | 0.7900 | 0.7900 | 0.7907 | 0.7905 | +0.0000 [-0.0030, +0.0025] | +0.0005 [-0.0015, +0.0025] | -0.0002 [-0.0036, +0.0025] |

## Final-refit routing composition

Counts and variance fractions below are averages across five folds; basis indices are fold-local.

| Task | C+ / fold | C- / fold | C0 / fold | mean r(C+) | mean r(C-) | C+ variance fraction | C- variance fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | 6.40 | 1.20 | 8.40 | 1.2445 | 0.9415 | 0.0900 | 0.0204 |
| OpenBMI_ERP | 14.20 | 0.60 | 1.20 | 1.1909 | 0.9689 | 0.3024 | 0.0102 |
| OpenBMI_SSVEP | 11.20 | 0.00 | 4.80 | 1.1054 | 1.0000 | 0.2457 | 0.0000 |
| WBCIC_MI | 5.00 | 2.60 | 8.40 | 1.1661 | 0.9249 | 0.5896 | 0.1991 |

## Paired biological-subject bootstrap contrasts

20,000 resamples within task; unit is the biological subject. Differences are UGCR-V3 minus comparator. Macro-F1, worst-session BA, and NLL are also in `PAIRED_HELDOUT_CONTRASTS.csv`.

| Task | Comparator | Δ BA [95% CI] | Δ Macro-F1 [95% CI] | Δ worst-session BA [95% CI] | Δ NLL |
|---|---|---:|---:|---:|---:|
| OpenBMI_MI | BASELINE | -0.0036 [-0.0121, +0.0057] | -0.0040 [-0.0128, +0.0055] | -0.0029 [-0.0093, +0.0029] | +0.0047 |
| OpenBMI_MI | POSITIVE_ONLY | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 |
| OpenBMI_MI | SHUFFLED | -0.0007 [-0.0079, +0.0079] | -0.0009 [-0.0085, +0.0075] | -0.0036 [-0.0079, +0.0000] | +0.0037 |
| OpenBMI_MI | RANDOM_MEAN | -0.0043 [-0.0143, +0.0057] | -0.0047 [-0.0143, +0.0052] | -0.0027 [-0.0093, +0.0031] | +0.0047 |
| OpenBMI_MI | BEST_RANDOM | -0.0043 [-0.0143, +0.0057] | -0.0047 [-0.0145, +0.0052] | -0.0021 [-0.0086, +0.0043] | +0.0047 |
| OpenBMI_ERP | BASELINE | -0.0021 [-0.0048, +0.0002] | +0.0060 [+0.0030, +0.0087] | -0.0031 [-0.0063, -0.0003] | -0.0117 |
| OpenBMI_ERP | POSITIVE_ONLY | -0.0000 [-0.0001, +0.0000] | -0.0001 [-0.0002, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 |
| OpenBMI_ERP | SHUFFLED | +0.0008 [-0.0000, +0.0016] | +0.0022 [+0.0014, +0.0032] | +0.0001 [-0.0015, +0.0014] | -0.0028 |
| OpenBMI_ERP | RANDOM_MEAN | -0.0020 [-0.0048, +0.0003] | +0.0058 [+0.0028, +0.0086] | -0.0032 [-0.0063, -0.0004] | -0.0114 |
| OpenBMI_ERP | BEST_RANDOM | -0.0024 [-0.0052, -0.0001] | +0.0055 [+0.0025, +0.0082] | -0.0033 [-0.0063, -0.0006] | -0.0114 |
| OpenBMI_SSVEP | BASELINE | +0.0043 [-0.0007, +0.0107] | +0.0042 [-0.0007, +0.0105] | +0.0043 [-0.0014, +0.0114] | +0.0038 |
| OpenBMI_SSVEP | POSITIVE_ONLY | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 |
| OpenBMI_SSVEP | SHUFFLED | +0.0036 [-0.0014, +0.0100] | +0.0035 [-0.0007, +0.0098] | +0.0050 [-0.0021, +0.0136] | +0.0037 |
| OpenBMI_SSVEP | RANDOM_MEAN | +0.0043 [-0.0007, +0.0107] | +0.0042 [-0.0007, +0.0105] | +0.0043 [-0.0014, +0.0114] | +0.0038 |
| OpenBMI_SSVEP | BEST_RANDOM | +0.0043 [-0.0007, +0.0107] | +0.0042 [-0.0007, +0.0105] | +0.0043 [-0.0014, +0.0114] | +0.0037 |
| WBCIC_MI | BASELINE | +0.0000 [-0.0030, +0.0025] | +0.0007 [-0.0026, +0.0035] | -0.0035 [-0.0085, +0.0015] | -0.0004 |
| WBCIC_MI | POSITIVE_ONLY | +0.0005 [+0.0000, +0.0015] | +0.0005 [+0.0000, +0.0016] | -0.0005 [-0.0015, +0.0000] | -0.0001 |
| WBCIC_MI | SHUFFLED | +0.0005 [-0.0015, +0.0025] | +0.0013 [-0.0012, +0.0035] | -0.0025 [-0.0075, +0.0020] | -0.0005 |
| WBCIC_MI | RANDOM_MEAN | -0.0002 [-0.0036, +0.0025] | +0.0005 [-0.0033, +0.0035] | -0.0036 [-0.0089, +0.0014] | -0.0004 |
| WBCIC_MI | BEST_RANDOM | -0.0005 [-0.0045, +0.0025] | +0.0001 [-0.0043, +0.0035] | -0.0040 [-0.0095, +0.0010] | -0.0004 |

## Interpretation

Overall state: **UTILITY_NOT_DIRECTION_SPECIFIC**. Per-task states: OpenBMI_MI=UTILITY_NOT_DIRECTION_SPECIFIC, OpenBMI_ERP=UTILITY_NOT_DIRECTION_SPECIFIC, OpenBMI_SSVEP=UTILITY_NOT_DIRECTION_SPECIFIC, WBCIC_MI=UTILITY_NOT_DIRECTION_SPECIFIC.
The state rules were fixed in the source protocol before heldout access. The best random draw is selected post-hoc by task BA among the five prelocked draws and is descriptive. A functional margin utility is not evidence of biological causality.

## Heldout diagnostics

`ROUTING_MECHANISM_AUDIT.csv` reports P movement and downstream C-preservation error, marked `POST_HOC_HELDOUT_DIAGNOSTIC`. `HELDOUT_RESCUE_HARM.csv` reports errors rescued and correct trials damaged relative to five-fold baseline probabilities, with C+/C-/high-utility activation energy on those trials. These are descriptive and did not change routing.

## Reproducibility

Protocol lock SHA-256: `a157d5871ab9c78c8011b57af0cf1720c2733a17a34c48fc20869a538fb1cad8`. Source commit: `80b582cd6f1fc94bbe04ed414dc303575028f542`. All nine variants use the same five-fold probability aggregation, and all baseline checkpoints match V1 and V2 byte-for-byte.
