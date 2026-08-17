# Paper-ready results

**Terminal state:** `FINAL_MULTIBACKBONE_FALSIFICATION_CLOSURE`

We prospectively evaluated four new EEG representation families—FBCNet,
EEGConformer, DeepConvNet and TeCh—against the frozen EEGNet reference using
the identical 41-subject development cohort, S1+S2→S3 protocol and four
persistence-rank blocks. Candidate actionability required persistence, harmful
signed utility, local and finite decision dependence, ≥0.5 percentage-point
specific balanced-accuracy gain, subject/fold stability, and Holm control over
the full 4×4 candidate family. No new backbone/block target jointly survived H1-H5 and prospective global multiplicity. The frozen five-backbone search is closed; AGDI is not authorized. The ten held-out
subjects were not opened.

| Backbone     | Competence   |   Best_task_BA |   Task_BA_CI_L |   Task_BA_CI_U |   Representation_dimension |   Audit_baseline_BA |   Persistent_blocks |   Protected_blocks |   Harmful_utility_blocks |   Decision_active_harmful_blocks |   Actionable_harmful_blocks | Final_recommended_action   | AGDI_authorized   | AGDI_dev_result   | Outer_result   |
|:-------------|:-------------|---------------:|---------------:|---------------:|---------------------------:|--------------------:|--------------------:|-------------------:|-------------------------:|---------------------------------:|----------------------------:|:---------------------------|:------------------|:------------------|:---------------|
| EEGNet       | True         |       0.794355 |       0.74878  |       0.838274 |                         32 |            0.781893 |                   2 |                  1 |                        1 |                                0 |                           0 | PRESERVE                   | False             | NOT_AUTHORIZED    | SEALED_UNUSED  |
| FBCNet       | False        |       0.503782 |       0.498782 |       0.509513 |                        288 |          nan        |                   0 |                  0 |                        0 |                                0 |                           0 | NO_OP                      | False             | NOT_AUTHORIZED    | SEALED_UNUSED  |
| EEGConformer | True         |       0.609674 |       0.579024 |       0.642156 |                         64 |            0.569416 |                   2 |                  0 |                        0 |                                0 |                           0 | NO_OP                      | False             | NOT_AUTHORIZED    | SEALED_UNUSED  |
| DeepConvNet  | True         |       0.76884  |       0.721462 |       0.813354 |                        256 |            0.738079 |                   2 |                  1 |                        0 |                                0 |                           0 | PRESERVE                   | False             | NOT_AUTHORIZED    | SEALED_UNUSED  |
| TeCh         | True         |       0.712966 |       0.668369 |       0.757274 |                         64 |            0.683543 |                   1 |                  1 |                        0 |                                0 |                           0 | PRESERVE                   | False             | NOT_AUTHORIZED    | SEALED_UNUSED  |
