# Rescue-first native SIRE P distillation: completed seed-0 pilot

Canonical SIRE-EEG, OpenBMI MI, folds 0–4, 20 fixed epochs. Only `depth1.weight` and `point1.weight` were trained. No CE, KD, margin or classifier loss was used. Geometry, normalizer, suffix and all BatchNorm state were frozen. All results use subject-disjoint discovery; discovery labels entered only the diagnostic teacher after the epoch-20 checkpoint and native evaluation were locked. This run read zero outer-dev and final-heldout EEG arrays. The historical canonical checkpoints had prior final-heldout diagnostic exposure, as disclosed in the source record.

## Q1. Rescue-first teacher headroom

Training native errors: 2896; single-direction prediction rescues: 636; coverage 22.0%. Correct native trials receive native P without search. Unrescued wrong trials also preserve native P. The teacher chooses minimum standardized P movement among *actual prediction rescues*, with deterministic CE/alpha/direction tie-breaks.

| Fold | Native wrong | Rescued | Coverage | Rescue P movement | Rescue CE improvement | Selected alpha counts | Selected direction counts |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 790 | 132 | 16.7% | 0.0723 | 0.0978 | {"0.0": 21, "0.25": 30, "0.5": 13, "0.75": 15, "1.25": 5, "1.5": 12, "2.0": 36} | {"0": 7, "1": 9, "2": 63, "3": 1, "5": 4, "7": 1, "8": 8, "9": 3, "11": 9, "12": 22, "13": 3, "15": 2} |
| 1 | 575 | 124 | 21.6% | 0.0628 | 0.1216 | {"0.0": 22, "0.25": 25, "0.5": 12, "0.75": 5, "1.25": 6, "1.5": 11, "2.0": 43} | {"0": 31, "1": 1, "2": 29, "3": 3, "4": 6, "5": 2, "6": 13, "7": 2, "10": 1, "14": 3, "15": 33} |
| 2 | 478 | 184 | 38.5% | 0.0676 | 0.1947 | {"0.0": 23, "0.25": 27, "0.5": 31, "0.75": 15, "1.25": 14, "1.5": 26, "2.0": 48} | {"0": 113, "3": 14, "4": 2, "5": 6, "8": 9, "10": 2, "12": 2, "13": 34, "14": 1, "15": 1} |
| 3 | 510 | 115 | 22.5% | 0.0595 | 0.1197 | {"0.0": 25, "0.25": 13, "0.5": 13, "0.75": 4, "1.25": 7, "1.5": 14, "2.0": 39} | {"1": 9, "3": 2, "4": 1, "5": 5, "6": 12, "7": 2, "8": 23, "9": 31, "10": 16, "11": 1, "12": 11, "14": 1, "15": 1} |
| 4 | 543 | 81 | 14.9% | 0.0462 | 0.0872 | {"0.0": 22, "0.25": 9, "0.5": 6, "0.75": 5, "1.25": 10, "1.5": 8, "2.0": 21} | {"0": 7, "1": 41, "4": 1, "5": 1, "6": 1, "7": 2, "8": 10, "10": 2, "11": 5, "12": 3, "14": 8} |

The 22.0% training rescue coverage exceeds the locked 5% headroom threshold, so teacher scarcity alone does not explain a failed student result.

## Q2. Did the student learn rescue ΔP?

All update statistics below are restricted to discovery trials whose frozen single-direction teacher actually rescued the native prediction. `E_zero` is the no-update error, and values of `E_student/E_zero` below 1 favor the student. The training loss gives RESCUE and PRESERVE their own group means; samples were neither duplicated nor oversampled.

| Fold | Rescue trials | E_student/E_zero | Cosine | Coordinate sign agreement | Update norm ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 36 | 1.118 | -0.039 | 0.486 | 0.687 |
| 1 | 36 | 1.132 | -0.179 | 0.420 | 0.896 |
| 2 | 39 | 1.023 | 0.189 | 0.606 | 0.551 |
| 3 | 24 | 1.029 | -0.083 | 0.505 | 1.540 |
| 4 | 18 | 0.983 | 0.092 | 0.486 | 0.746 |
| Pooled | 153 | 1.059 | -0.005 | 0.504 | 0.842 |

| Training diagnostic | Epoch 0 | Epoch 20 |
| --- | ---: | ---: |
| L_P_rescue | 0.01148 | 0.00482 |
| L_P_preserve | 0.00000 | 0.00053 |
| L_C | 0.00000 | 0.00114 |
| P_drift_standardized_RMS | 0.00000 | 0.01880 |
| C_drift_standardized_RMS | 0.00000 | 0.03000 |

The full 0–20 curves and the number of RESCUE samples seen each epoch are in `TRAINING_HISTORY.csv`.
The train rescue-target error fell, but discovery rescue-target update error remained 1.059 times the zero-update error. Its near-zero pooled cosine provides no evidence that the student generalized the required P direction.

## Q3. Native classifier performance

Per fold and pooled BA below are biological-subject equal, averaging both discovery sessions within subject and repeated fold appearances within biological subject. CIs use 20,000 paired biological-subject bootstrap draws. The frozen prior experiment's CE-only and Local-P results are historical context only and were not used for training or selection.

| Fold | Baseline BA | Rescue-first BA | Difference | Paired 95% CI |
| --- | ---: | ---: | ---: | --- |
| 0 | 0.7317 | 0.7317 | +0.0000 | [-0.0058, +0.0058] |
| 1 | 0.8050 | 0.8025 | -0.0025 | [-0.0083, +0.0017] |
| 2 | 0.7942 | 0.7942 | -0.0000 | [-0.0042, +0.0067] |
| 3 | 0.7783 | 0.7667 | -0.0117 | [-0.0217, -0.0058] |
| 4 | 0.7392 | 0.7350 | -0.0042 | [-0.0108, +0.0008] |
| Pooled subjects | 0.7738 | 0.7705 | -0.0033 | [-0.0064, +0.0000] |

| Arm | Macro-F1 | NLL | Worst-session BA | Future-session BA |
| --- | ---: | ---: | ---: | ---: |
| BASELINE | 0.7680 | 0.5543 | 0.7417 | 0.7690 |
| RESCUE_FIRST_P | 0.7645 | 0.5545 | 0.7361 | 0.7647 |

| Pooled paired metric | Rescue-first minus baseline | Paired 95% CI |
| --- | ---: | --- |
| BA | -0.0033 | [-0.0064, +0.0000] |
| macro_F1 | -0.0034 | [-0.0067, -0.0000] |
| NLL | +0.0002 | [-0.0028, +0.0039] |
| worst_session_BA | -0.0057 | [-0.0098, -0.0015] |
| future_session_BA | -0.0043 | [-0.0086, +0.0000] |

The pooled BA difference is -0.0033; the fixed endpoint does not outperform the original checkpoint.

## Q4. Prediction transitions

The following counts are per evaluated discovery trial across five folds; a biological subject can appear in multiple folds. Native-wrong teacher coverage on discovery is 153/1382 native errors. Student wrong→correct: 28; correct→wrong: 50.

| Teacher group | Trials | Correct→correct | Correct→wrong | Wrong→correct | Wrong→wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| NATIVE_CORRECT | 4618 | 4568 | 50 | 0 | 0 |
| TEACHER_RESCUABLE_ERROR | 153 | 0 | 0 | 26 | 127 |
| TEACHER_UNRESCUABLE_ERROR | 1229 | 0 | 0 | 2 | 1227 |

Every discovery trial's prediction and target type is in `PREDICTION_TRANSITION_AUDIT.csv`.
The student corrected 28/1382 native errors (2.0%) while damaging 50 native-correct predictions. Among 153 teacher-rescuable discovery errors it corrected 26 (17.0%).

## Q5. P/C transplant attribution

All representations use the same original frozen suffix. P-only and C-only are diagnostic transplants, so their effects need not add.

| Representation | Subject-equal BA | Native errors rescued | Rescue rate |
| --- | ---: | ---: | ---: |
| BASE | 0.7738 | 0/1382 | 0.0% |
| FULL_STUDENT | 0.7705 | 28/1382 | 2.0% |
| P_ONLY_CHANGE | 0.7723 | 17/1382 | 1.2% |
| C_ONLY_CHANGE | 0.7732 | 16/1382 | 1.2% |

P-only/full BA-gain retention: undefined because FULL_STUDENT does not exceed BASE. FULL minus BASE BA: -0.0033; P-only minus BASE BA: -0.0015.
The full student has no positive gain to attribute to P. The P-only transplant also falls below BASE, so the experiment provides no P-mediated classifier improvement.

## Q6. P/C drift

| Discovery representation movement | Mean |
| --- | ---: |
| P_raw_L2_movement_mean | 0.1763 |
| P_standardized_RMS_movement_mean | 0.0194 |
| C_raw_L2_movement_mean | 1.6722 |
| C_standardized_RMS_movement_mean | 0.0301 |

The small standardized P and C movements show that this training objective constrained overall representation drift, despite failing to transfer the rescue direction to discovery.

## Diagnostic classification

`RESCUE_TARGET_NOT_DISTILLABLE`. This label is a development diagnostic rather than independent validation. The fixed headroom threshold is 5% of training native errors. P-only attribution is required for the positive label. Full parameter/BN state, teacher-cache, exact historical geometry-hash, batch exposure and leakage audits are included in the compact outputs and protocol. Dense teacher caches and epoch-20 checkpoints remain in server runtime storage.
