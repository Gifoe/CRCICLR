# Native SIRE local P construction: completed seed-0 pilot

Canonical SIRE-EEG / CompactLite, OpenBMI MI, folds 0–4. Only `depth1.weight` and `point1.weight` were trainable; all BN states, dropout behavior, source branches and suffix remained frozen. Epoch 20 is the fixed endpoint. The primary aggregate averages the two discovery sessions within each biological subject, then averages subjects, with repeated fold appearances deduplicated. Future-session BA is also shown. All confidence intervals use 20,000 paired biological-subject bootstrap draws.

The canonical checkpoints have historical final-heldout diagnostic exposure in the source record. This run read zero outer-dev or final-heldout EEG arrays and used no discovery label for training or selection.

## Q1. Performance against CE-only continuation

| Fold | Baseline BA | CE-only BA | Local-P BA | Local-P minus CE-only | 95% paired CI |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 0.7317 | 0.7058 | 0.7133 | +0.0075 | [-0.0042, +0.0183] |
| 1 | 0.8050 | 0.7875 | 0.7958 | +0.0083 | [-0.0008, +0.0175] |
| 2 | 0.7942 | 0.7833 | 0.7883 | +0.0050 | [-0.0058, +0.0167] |
| 3 | 0.7783 | 0.7717 | 0.7750 | +0.0033 | [-0.0075, +0.0117] |
| 4 | 0.7392 | 0.7317 | 0.7317 | -0.0000 | [-0.0058, +0.0058] |
| Pooled biological subjects | 0.7738 | 0.7591 | 0.7652 | +0.0061 | [+0.0012, +0.0110] |

| Arm | Macro-F1 | NLL | Worst-session BA | Future-session BA |
| --- | ---: | ---: | ---: | ---: |
| BASELINE | 0.7680 | 0.5543 | 0.7417 | 0.7690 |
| CE_ONLY_CONTINUATION | 0.7513 | 0.5757 | 0.7270 | 0.7464 |
| LOCAL_CONSTRAINED_P | 0.7578 | 0.5664 | 0.7348 | 0.7533 |

The comparison with the original checkpoint is secondary. The complete per-subject/session values and all paired metric contrasts are in `DISCOVERY_METRICS.csv`, `SUBJECT_METRICS.csv`, and `PAIRED_CONTRASTS.csv`.

## Q2. Usable local teacher supervision

The teacher searches one source-C direction at a time over alpha 0.5, 0.75, 1.25 and 1.5, accepts only a CE improvement greater than 1e-4 without damaging a native-correct prediction, and selects the smallest standardized P movement. It is generated on inner-train labels using the original frozen shared block. Accepted 24799/26000 training trials (95.4%).

| Fold | Accepted | Mean accepted CE improvement | Mean accepted P movement | Selected alpha counts | Selected direction counts |
| --- | ---: | ---: | ---: | --- | --- |
| 0 | 97.7% | 0.0004 | 0.0028 | {"0.5": 530, "0.75": 1941, "1.25": 1996, "1.5": 614} | {"0": 46, "1": 1521, "2": 501, "3": 86, "4": 60, "5": 262, "6": 100, "7": 102, "8": 396, "9": 340, "10": 45, "11": 459, "12": 825, "13": 113, "14": 61, "15": 164} |
| 1 | 97.8% | 0.0003 | 0.0043 | {"0.5": 683, "0.75": 1869, "1.25": 1838, "1.5": 693} | {"0": 249, "1": 295, "2": 378, "3": 255, "4": 189, "5": 104, "6": 870, "7": 339, "8": 157, "9": 56, "10": 252, "11": 291, "12": 61, "13": 297, "14": 304, "15": 986} |
| 2 | 96.5% | 0.0004 | 0.0047 | {"0.5": 664, "0.75": 1923, "1.25": 1796, "1.5": 637} | {"0": 459, "1": 102, "2": 161, "3": 417, "4": 410, "5": 194, "6": 86, "7": 67, "8": 596, "9": 198, "10": 347, "11": 266, "12": 403, "13": 917, "14": 131, "15": 266} |
| 3 | 94.1% | 0.0004 | 0.0050 | {"0.5": 614, "0.75": 1833, "1.25": 1804, "1.5": 644} | {"0": 299, "1": 115, "2": 23, "3": 161, "4": 92, "5": 380, "6": 764, "7": 103, "8": 613, "9": 567, "10": 392, "11": 175, "12": 466, "13": 231, "14": 105, "15": 409} |
| 4 | 90.8% | 0.0003 | 0.0049 | {"0.5": 626, "0.75": 1671, "1.25": 1718, "1.5": 705} | {"0": 993, "1": 614, "2": 18, "3": 96, "4": 101, "5": 193, "6": 131, "7": 311, "8": 834, "9": 124, "10": 181, "11": 286, "12": 160, "13": 25, "14": 243, "15": 410} |

## Q3. Learned P update

`E_zero` is the squared standardized error of making no P update; `E_student` is the error of the native student P update. Values below 1 favor the student. Cosines use nonzero update pairs and are post-hoc diagnostics only.

| Fold | E_student / E_zero | Mean cosine | Better than zero update |
| --- | ---: | ---: | --- |
| 0 | 18.887 | -0.006 | False |
| 1 | 12.722 | +0.016 | False |
| 2 | 15.194 | +0.028 | False |
| 3 | 21.975 | +0.001 | False |
| 4 | 17.362 | +0.031 | False |
| Pooled | 17.888 | +0.014 | False |

## Q4. P/C transplant attribution

All four representations pass through the same frozen suffix. These evaluations are diagnostic and need not be additive.

| Fold | BASE BA | FULL_STUDENT BA | P_ONLY_CHANGE BA | C_ONLY_CHANGE BA |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.7317 | 0.7133 | 0.7317 | 0.7167 |
| 1 | 0.8050 | 0.7958 | 0.8033 | 0.7983 |
| 2 | 0.7942 | 0.7883 | 0.7925 | 0.7933 |
| 3 | 0.7783 | 0.7750 | 0.7725 | 0.7742 |
| 4 | 0.7392 | 0.7317 | 0.7392 | 0.7300 |
| Pooled | 0.7738 | 0.7652 | 0.7725 | 0.7673 |

P-only/full gain ratio: undefined because FULL_STUDENT does not beat BASE. FULL minus BASE BA: -0.0086; P-only minus BASE BA: -0.0013. A positive classifier result is P mediated only when the P-only transplant retains a substantial part of the full gain.

## Q5. Native P/C drift

| Arm | Standardized P drift | Standardized C drift | P drift/native norm | C drift/native norm |
| --- | ---: | ---: | ---: | ---: |
| CE_ONLY_CONTINUATION | 0.1444 | 0.2009 | 0.1828 | 0.2049 |
| LOCAL_CONSTRAINED_P | 0.0292 | 0.0732 | 0.0370 | 0.0749 |

## Diagnostic classification

`PERFORMANCE_GAIN_NOT_P_MEDIATED`. Local-P minus CE-only BA is +0.0061; 4/5 folds have a positive difference; P-update error ratio is 17.888. The label is a development diagnostic, not an independent validation claim.

The exact old SIRE geometry hashes, teacher cache hashes, trainable-state whitelist, BN byte identity and identical batch/optimizer-step audits are recorded in the protocol and compact outputs. Trial caches and checkpoints remain in server runtime storage.
