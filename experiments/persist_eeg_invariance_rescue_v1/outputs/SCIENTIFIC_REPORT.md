# Experiment 1 scientific report

Status: `NO_ELIGIBLE_PROTECTED_LOSS_OBSERVED`. All results are **DEVELOPMENT / EXPLORATORY**.

## Direct answers

1. Methods with measurable mean subject-information reduction: `['C_SCLDGN']`.
2. Families passing the frozen all-run protected-loss gate I2 after I1: `[]`. C has PRS defined in only `4/6` runs, so its large observed PRS decrease is incomplete rather than evidence of preservation.
3. Families where protected loss accompanied future-session task harm: `[]`.
4. No rescue was fit or scored because no family passed I1+I2+I3.
5. Not estimable: Generic and PERSIST rescue were not run, so no paired rescue CI exists.
6. Not estimable: no eligible family entered rescue, hence no post-rescue identity probe exists.
7. Evidence spans `3` folds, `2` seeds, and the subject counts in the statistics tables. Cross-family support: `False`.
8. Strongest counterexample to a blanket invariance-harm claim: `C_SCLDGN`. C reduced identity but had only `4/6` Protected assignments and DeltaBA CI `[-0.0296296296296296, 0.03499999999999997]`.
9. Outer test used: `false`.
10. Terminal state: `NO_ELIGIBLE_PROTECTED_LOSS_OBSERVED`.

Protected-retention task-only R2 is expected to approach one because its target is a frozen linear coordinate of the same task-only representation. The evidential quantity is the cross-subject, cross-session recoverability change in the independently trained invariant representation, not task-only self-reconstruction in isolation.

## Strict diagnostic interpretation

- The primary GRL lambda 0.3 did not reduce identity on average (`DeltaID=0.0046`). The lower 0.05/0.10 ladder points did reduce mean identity, but promoting either after seeing outcome would be post hoc selection. Full ladder means (lambda encoded in thousandths) are: 0050: DeltaID=-0.0306, DeltaPRS=-0.6465, DeltaBA=-0.0020; 0100: DeltaID=-0.0302, DeltaPRS=-0.6644, DeltaBA=-0.0091; 0300: DeltaID=0.0046, DeltaPRS=-0.6499, DeltaBA=-0.0094; 1000: DeltaID=0.0093, DeltaPRS=-0.6414, DeltaBA=-0.0096.
- The clean-room EEG-DG instantiation harmed task BA but increased rather than reduced the cross-session identity probe (`DeltaID=0.0419`). Its task harm therefore cannot support the proposed invariance-to-protected-loss causal chain.
- SCLDGN produced the only certified identity reduction, but the mean task effect was negligible and non-certified (`DeltaBA=-0.0014814814814814996`, CI `[-0.0296296296296296, 0.03499999999999997]`), while Protected assignment was absent in two runs. The formal label `INVARIANCE_PRESERVES_PROTECTED_STRUCTURE` means only that frozen I2 failed; it must not be read as affirmative preservation evidence.
- PRS is not protected-selectivity proof by itself. For A, invariant protected/non-Protected R2 was `0.3501/0.3552`; for C it was `0.0937/0.0801` over evaluable runs. These similar drops are compatible with broad cross-model linear non-identifiability. B showed a larger protected-specific gap, but failed I1.

## Table 1 — Invariance audit

| family | task_only_BA | invariant_BA | delta_BA_INV | task_only_subject_probe | invariant_subject_probe | delta_ID | task_only_protected_retention | invariant_protected_retention | delta_PRS | task_only_matched_nonprotected_retention | invariant_matched_nonprotected_retention | I1 | I2 | I3 | eligibility_status | protected_assignment_runs | runs | outer_test_used | delta_ID_CI95_low | delta_ID_CI95_high | delta_ID_raw_p_less | delta_ID_holm_p_less | delta_PRS_CI95_low | delta_PRS_CI95_high | delta_PRS_raw_p_less | delta_PRS_holm_p_less | delta_BA_INV_CI95_low | delta_BA_INV_CI95_high | delta_BA_INV_raw_p_less | delta_BA_INV_holm_p_less |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_SUBJECT_GRL_EEGNET | 0.7063 | 0.6969 | -0.0094 | 0.3561 | 0.3607 | 0.0046 | 1.0000 | 0.3501 | -0.6499 | 1.0000 | 0.3552 | False | True | True | NO_MEASURABLE_INVARIANCE_EFFECT | 6 | 6 | False | -0.0620 | 0.0787 | 0.6507 | 1.0000 | -0.7365 | -0.5816 | 0.0000 | 0.0000 | -0.0317 | 0.0120 | 0.1853 | 0.3705 |
| B_EEG_DG | 0.7228 | 0.6467 | -0.0761 | 0.3211 | 0.3630 | 0.0419 | 1.0000 | 0.1135 | -0.8865 | 1.0000 | 0.3273 | False | True | True | NO_MEASURABLE_INVARIANCE_EFFECT | 6 | 6 | False | -0.0519 | 0.1282 | 0.9006 | 1.0000 | -0.9637 | -0.8319 | 0.0000 | 0.0000 | -0.1459 | -0.0341 | 0.0000 | 0.0000 |
| C_SCLDGN | 0.7194 | 0.7180 | -0.0015 | 0.3748 | 0.2441 | -0.1307 | 1.0000 | 0.0937 | -0.9063 | 1.0000 | 0.0801 | True | False | True | INVARIANCE_PRESERVES_PROTECTED_STRUCTURE | 4 | 6 | False | -0.1861 | -0.0689 | 0.0000 | 0.0000 | -0.9369 | -0.8654 | 0.0000 | 0.0000 | -0.0296 | 0.0350 | 0.3232 | 0.3705 |

## Table 2 — Rescue

No eligible family; rescue table is empty by protocol.

## Table 3 — Attribution

No eligible family; attribution table is empty by protocol.

## Interpretation boundary

This experiment does not establish that subject invariance is generally wrong, does not test absolute SOTA, and does not authorize an outer-test claim. It only tests whether the preregistered independently trained invariance objectives exhibit the full identity-loss/protected-loss/task-harm chain and, if so, whether intervention-defined protected restoration beats matched controls.

It is insufficient by itself to justify Experiment 2 as a PERSIST-rescue follow-up. Proceed only if Experiment 2 is framed independently and first repairs the prerequisites: a reproducibly identity-reducing primary method, complete Protected assignment, and a retention metric that separates protected loss from generic cross-model alignability.

## Provenance and run accounting

- `RUN_LEDGER.csv` contains `54` full science runs, `9` binding smoke runs, and `3` excluded pre-freeze debug attempts. `RUN_LEDGER_FULL.csv` preserves the 54-row full-run view.
- The complete `SPLIT_FREEZE.json` was SHA-256 hashed as opaque file bytes for provenance. Only `train_subjects` and `validation_subjects` were indexed; outer membership was not extracted, enumerated, logged, featurized, or scored.
