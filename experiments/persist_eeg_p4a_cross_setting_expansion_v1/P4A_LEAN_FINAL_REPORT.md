# P4A Lean Final Report

Exact terminal: `P4A_LEAN_CROSS_SETTING_CUBE_COMPLETE`.

## Competent-ERM settings

| setting_id   | dataset   | task   | backbone     |   subject_count |   folds |   seeds |   outcome_BA_mean |   outcome_macro_F1_mean |   folds_above_chance |   representation_dim | session_roles                         | source_artifact_complete   | preprocessing_event_label_status   | competence      |
|:-------------|:----------|:-------|:-------------|----------------:|--------:|--------:|------------------:|------------------------:|---------------------:|---------------------:|:--------------------------------------|:---------------------------|:-----------------------------------|:----------------|
| S1           | OpenBMI   | MI     | EEGNet       |              40 |       5 |       3 |           0.75492 |                 0.75127 |                    5 |                   64 | source S1+S2; outcome held-subject S2 | True                       | PASS                               | COMPETENCE_PASS |
| S2           | OpenBMI   | MI     | EEGConformer |              40 |       5 |       3 |           0.77192 |                 0.76630 |                    5 |                   64 | source S1+S2; outcome held-subject S2 | True                       | PASS                               | COMPETENCE_PASS |
| S3           | WBCIC     | MI     | EEGNet       |              41 |       5 |       3 |           0.78843 |                 0.78708 |                    5 |                   32 | source S1+S2; outcome held-subject S3 | True                       | PASS                               | COMPETENCE_PASS |
| S4           | WBCIC     | MI     | EEGConformer |              41 |       5 |       3 |           0.78441 |                 0.78261 |                    5 |                   64 | source S1+S2; outcome held-subject S3 | True                       | PASS                               | COMPETENCE_PASS |
| S5           | OpenBMI   | ERP    | EEGNet       |              40 |       5 |       3 |           0.85635 |                 0.80856 |                    5 |                   64 | source S1+S2; outcome held-subject S2 | True                       | PASS                               | COMPETENCE_PASS |
| S6           | OpenBMI   | ERP    | EEGConformer |              40 |       5 |       3 |           0.83390 |                 0.78107 |                    5 |                   64 | source S1+S2; outcome held-subject S2 | True                       | PASS                               | COMPETENCE_PASS |

## Lean closure

- Mandatory new-setting ERM: 45/45 complete before amendment.
- Cross-setting ERM cube: 90 rows (6 settings × 5 folds × 3 seeds).
- Source evidence cube: 720 rows; first eight source-defined persistent directions per run.
- Source artifact audit: 90 complete run rows.
- Usable settings: S1, S2, S3, S4, S5, S6.
- Failed settings: none.
- Non-ERM grid: partial and explicitly excluded from the Lean primary gate; see pause snapshot.
- Source I/P/D/C_src/O_task definitions: unchanged from the frozen P4A protocol.
- O_task: squared projection on the centered frozen linear-head task span; no future-driven replacement.
- New-setting direction-level future utility: sealed.
- Invariance outcome deltas: sealed.
- OpenBMI sealed 14: untouched and unenumerated.
- WBCIC outer 10: untouched and unenumerated.

This closure establishes a competent-ERM, cross-dataset, cross-task, and cross-backbone source evidence cube. It does not claim completion of the original 405-grid protocol and does not use partial-grid outcomes to formulate P4B.
