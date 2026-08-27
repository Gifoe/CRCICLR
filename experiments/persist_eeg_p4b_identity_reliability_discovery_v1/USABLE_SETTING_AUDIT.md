# Usable Setting Audit

Eligibility requires P4A `COMPETENCE_PASS`, complete 15-run source artifacts, and preprocessing/event/label PASS. No retuning was performed.

| setting_id   | dataset   | task   | backbone     |   subject_count |   folds |   seeds |   outcome_BA_mean |   outcome_macro_F1_mean |   folds_above_chance |   representation_dim | session_roles                         | source_artifact_complete   | preprocessing_event_label_status   | competence      |
|:-------------|:----------|:-------|:-------------|----------------:|--------:|--------:|------------------:|------------------------:|---------------------:|---------------------:|:--------------------------------------|:---------------------------|:-----------------------------------|:----------------|
| S1           | OpenBMI   | MI     | EEGNet       |              40 |       5 |       3 |           0.75492 |                 0.75127 |                    5 |                   64 | source S1+S2; outcome held-subject S2 | True                       | PASS                               | COMPETENCE_PASS |
| S2           | OpenBMI   | MI     | EEGConformer |              40 |       5 |       3 |           0.77192 |                 0.76630 |                    5 |                   64 | source S1+S2; outcome held-subject S2 | True                       | PASS                               | COMPETENCE_PASS |
| S3           | WBCIC     | MI     | EEGNet       |              41 |       5 |       3 |           0.78843 |                 0.78708 |                    5 |                   32 | source S1+S2; outcome held-subject S3 | True                       | PASS                               | COMPETENCE_PASS |
| S4           | WBCIC     | MI     | EEGConformer |              41 |       5 |       3 |           0.78441 |                 0.78261 |                    5 |                   64 | source S1+S2; outcome held-subject S3 | True                       | PASS                               | COMPETENCE_PASS |
| S5           | OpenBMI   | ERP    | EEGNet       |              40 |       5 |       3 |           0.85635 |                 0.80856 |                    5 |                   64 | source S1+S2; outcome held-subject S2 | True                       | PASS                               | COMPETENCE_PASS |
| S6           | OpenBMI   | ERP    | EEGConformer |              40 |       5 |       3 |           0.83390 |                 0.78107 |                    5 |                   64 | source S1+S2; outcome held-subject S2 | True                       | PASS                               | COMPETENCE_PASS |

Discovery settings: S1, S2, S3, S5. P4C reserved settings: S4, S6. The S5 choice follows the predeclared priority and used no future utility.
