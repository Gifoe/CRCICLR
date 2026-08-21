# Redesign ledger

| Change | Reason | Timing |
|---|---|---|
| Add T_replica | Estimate ordinary independent retraining mismatch | Before V1.1 outcome |
| Replace raw PRS primary gate with functional SPL | Remove latent-coordinate non-identifiability confound | Before V1.1 outcome |
| Select GRL strength by 3-fold train-subject CV | Avoid post-hoc λ=.05/.10 promotion | Before V1.1 outcome |
| Refit all final models on 34 train subjects | Repair V1 28-subject fit underuse | Before V1.1 outcome |
| Permit secondary rescue for STATUS_C | Avoid censoring functional restoration when BA is redundant | Before V1.1 outcome |
| Write explicit `rescue_allowed` gate in `ELIGIBILITY.json` | Post-freeze implementation repair: `rescue.py` consumed this field, but the eligibility writer omitted it; the scientific gate is unchanged | After full outcome, code-only bug fix; no outcome values used |
| Use subject-level audit for I1 bootstrap | `IDENTITY_AUDIT.csv` is run-level; the paired `SUBJECT_LEVEL_AUDIT.csv` is required by the frozen subject-level inference unit | After full outcome, reporting/implementation repair; no gate or data change |
| Vectorize hierarchical bootstrap implementation | Remove repeated pandas filtering inside each draw; fold→seed→subject resampling definition is unchanged | After full outcome, performance-only repair |
| Separate functional-retention and SPL inputs for figures; write subject-level result table directly | Finalizer previously assumed columns that belong to different frozen output tables | After full outcome, output plumbing repair; no scientific change |
| Add L_N summaries and raw/Holm p-value fields | Complete the predeclared reporting contract without changing I1/I2/I3 status gates | After full outcome, reporting-only repair |
