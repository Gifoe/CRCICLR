# DATA LEGALITY AUDIT

## PERSIST-CDE SEED-0 PILOT

PASS: the pilot uses only the frozen canonical roles and authorized development subjects.

- OpenBMI: 54 Stage-0-frozen subjects, S1+S2 source and S2 outcome, five frozen folds.
- WBCIC: 41 subjects from `DEVELOPMENT_SCOPE_LOCK.json`, S1+S2 source and S3 outcome, five frozen folds.
- `outer_subject_ids_present=false` was asserted from the WBCIC scope lock; the sealed outer 10 were not enumerated or opened.
- No outcome subject history was used for adaptation, normalization, epoch selection or fusion selection.
- The only pre-final outcome access was the mandated checkpoint-equivalence comparison of IDs, labels and probabilities; no outcome metric entered selection.

Observed fold role counts:

- OpenBMI fold 0: model_fit=34, discovery=9, outcome=11; initial_rows=6800, discovery_rows=900, refit_rows=8600, outcome_rows=1100
- OpenBMI fold 1: model_fit=34, discovery=9, outcome=11; initial_rows=6800, discovery_rows=900, refit_rows=8600, outcome_rows=1100
- OpenBMI fold 2: model_fit=34, discovery=9, outcome=11; initial_rows=6800, discovery_rows=900, refit_rows=8600, outcome_rows=1100
- OpenBMI fold 3: model_fit=34, discovery=9, outcome=11; initial_rows=6800, discovery_rows=900, refit_rows=8600, outcome_rows=1100
- OpenBMI fold 4: model_fit=35, discovery=9, outcome=10; initial_rows=7000, discovery_rows=900, refit_rows=8800, outcome_rows=1000
- WBCIC fold 0: model_fit=24, discovery=8, outcome=9; initial_rows=9596, discovery_rows=1600, refit_rows=12796, outcome_rows=1800
- WBCIC fold 1: model_fit=25, discovery=8, outcome=8; initial_rows=9997, discovery_rows=1600, refit_rows=13196, outcome_rows=1600
- WBCIC fold 2: model_fit=25, discovery=8, outcome=8; initial_rows=9999, discovery_rows=1595, refit_rows=13197, outcome_rows=1600
- WBCIC fold 3: model_fit=25, discovery=8, outcome=8; initial_rows=9999, discovery_rows=1600, refit_rows=13198, outcome_rows=1595
- WBCIC fold 4: model_fit=24, discovery=9, outcome=8; initial_rows=9597, discovery_rows=1800, refit_rows=13197, outcome_rows=1600
