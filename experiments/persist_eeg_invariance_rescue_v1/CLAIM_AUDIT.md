# Claim audit

- Terminal claim: `NO_ELIGIBLE_PROTECTED_LOSS_OBSERVED`.
- `outer_test_used=false`.
- `SPLIT_FREEZE.json` was parsed to index only `train_subjects` and `validation_subjects`; outer membership was never extracted, enumerated, logged, featurized, or scored. The complete file bytes were SHA-256 hashed solely for provenance, so the uninspected outer-field bytes necessarily contributed opaquely to the file-level digest. No outer signal or label was accessed.
- Protected assignment used model-fit subjects only.
- No rescue model or rescue hyperparameter selection was executed because no family was eligible. The dormant preregistered selector is calibration-Session-2-only.
- Development outcome labels were used only for final task scoring; outcome Session 1/2 identity labels were used only for the preregistered cross-session probe.
- B/C are clean-room method-level reproductions, not exact official-source reproductions; conclusions are limited to these audited instantiations.
- Task-only PRS is a self-coordinate recoverability reference and is not independent evidence.
- A/C matched non-Protected retention fell similarly to Protected retention, so PRS loss may reflect broad cross-model linear non-identifiability and cannot be called selective deletion.
- C's formal status label records failure of the all-run I2 gate; with only 4/6 Protected assignments it is not affirmative evidence that protected structure was preserved.
- The lower GRL ladder points are exploratory diagnostics and were not promoted to primary after outcome inspection.
- No V6/V7/V8 generic adaptation, router, Conformer blend, action bank, or meta-selector was used.
