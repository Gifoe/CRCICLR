# PERSIST-EEG final multi-backbone closure

## Outcome

`FINAL_MULTIBACKBONE_FALSIFICATION_CLOSURE`

No new backbone/block target jointly survived H1-H5 and prospective global multiplicity. The frozen five-backbone search is closed; AGDI is not authorized.

The result is prospective over exactly four new representation families and
four fixed rank blocks. The family-wise test uses all 16 slots, including
incompetent-backbone slots as p=1. The ten outer WBCIC subjects remain sealed
and unused. EEGNet was not rerun or reinterpreted.

Competent representations: `EEGNet, EEGConformer, DeepConvNet, TeCh`. Competence failures:
`FBCNet`. A failed backbone does
not contribute H1--H5 evidence.

## Backbone summary

| Backbone     | Competence   |   Best_task_BA |   Task_BA_CI_L |   Task_BA_CI_U |   Representation_dimension |   Audit_baseline_BA |   Persistent_blocks |   Protected_blocks |   Harmful_utility_blocks |   Decision_active_harmful_blocks |   Actionable_harmful_blocks | Final_recommended_action   | AGDI_authorized   | AGDI_dev_result   | Outer_result   |
|:-------------|:-------------|---------------:|---------------:|---------------:|---------------------------:|--------------------:|--------------------:|-------------------:|-------------------------:|---------------------------------:|----------------------------:|:---------------------------|:------------------|:------------------|:---------------|
| EEGNet       | True         |       0.794355 |       0.74878  |       0.838274 |                         32 |            0.781893 |                   2 |                  1 |                        1 |                                0 |                           0 | PRESERVE                   | False             | NOT_AUTHORIZED    | SEALED_UNUSED  |
| FBCNet       | False        |       0.503782 |       0.498782 |       0.509513 |                        288 |          nan        |                   0 |                  0 |                        0 |                                0 |                           0 | NO_OP                      | False             | NOT_AUTHORIZED    | SEALED_UNUSED  |
| EEGConformer | True         |       0.609674 |       0.579024 |       0.642156 |                         64 |            0.569416 |                   2 |                  0 |                        0 |                                0 |                           0 | NO_OP                      | False             | NOT_AUTHORIZED    | SEALED_UNUSED  |
| DeepConvNet  | True         |       0.76884  |       0.721462 |       0.813354 |                        256 |            0.738079 |                   2 |                  1 |                        0 |                                0 |                           0 | PRESERVE                   | False             | NOT_AUTHORIZED    | SEALED_UNUSED  |
| TeCh         | True         |       0.712966 |       0.668369 |       0.757274 |                         64 |            0.683543 |                   1 |                  1 |                        0 |                                0 |                           0 | PRESERVE                   | False             | NOT_AUTHORIZED    | SEALED_UNUSED  |

## Gate/action matrix

| Backbone     | Block   | H1    | H2    | H3    | H4    | H5    | Protected   | Globally_qualified_actionable   | Action   |
|:-------------|:--------|:------|:------|:------|:------|:------|:------------|:--------------------------------|:---------|
| EEGNet       | P01_04  | True  | False | True  | False | False | True        | False                           | PRESERVE |
| EEGNet       | P05_08  | True  | True  | False | False | False | False       | False                           | NO_OP    |
| EEGNet       | P09_16  | False | False | False | False | False | False       | False                           | NO_OP    |
| EEGNet       | P17_32  | False | False | False | False | False | False       | False                           | NO_OP    |
| FBCNet       | P01_04  | False | False | False | False | False | False       | False                           | NO_OP    |
| FBCNet       | P05_08  | False | False | False | False | False | False       | False                           | NO_OP    |
| FBCNet       | P09_16  | False | False | False | False | False | False       | False                           | NO_OP    |
| FBCNet       | P17_32  | False | False | False | False | False | False       | False                           | NO_OP    |
| EEGConformer | P01_04  | True  | False | True  | False | False | False       | False                           | NO_OP    |
| EEGConformer | P05_08  | True  | False | False | False | False | False       | False                           | NO_OP    |
| EEGConformer | P09_16  | False | False | False | False | False | False       | False                           | NO_OP    |
| EEGConformer | P17_32  | False | False | False | False | False | False       | False                           | NO_OP    |
| DeepConvNet  | P01_04  | True  | False | True  | False | False | True        | False                           | PRESERVE |
| DeepConvNet  | P05_08  | True  | False | True  | False | False | False       | False                           | NO_OP    |
| DeepConvNet  | P09_16  | False | False | False | False | False | False       | False                           | NO_OP    |
| DeepConvNet  | P17_32  | False | False | False | False | False | False       | False                           | NO_OP    |
| TeCh         | P01_04  | True  | False | True  | False | False | True        | False                           | PRESERVE |
| TeCh         | P05_08  | False | False | False | False | True  | False       | False                           | NO_OP    |
| TeCh         | P09_16  | False | False | False | False | False | False       | False                           | NO_OP    |
| TeCh         | P17_32  | False | False | False | False | False | False       | False                           | NO_OP    |

## Interpretation limits

- A competence failure is not evidence that persistent nuisance is absent; it
  means that representation cannot support an interpretable PERSIST audit.
- The spectral FBCNet family failed near chance, so the closure contains no
  competent filter-bank audit. This weakens inductive-bias coverage and must
  not be described as five competent backbones.
- `EEGConformer` had a
  cross-fitted audit baseline below 0.60 after restricting training to the
  three model-fit folds; its block-level audit is correspondingly weaker than
  its task-search competence result.
- A negative H4/H5 result is evidence against safe removability under this
  frozen intervention, not proof that the block contains no subject signal.
- The study tests four pre-registered blocks in five representation families;
  it does not support claims about every possible architecture or latent basis.
- No theorem guarantees that removing a harmful-utility direction improves
  generalization. That empirical claim is exactly what H4/H5 test.
