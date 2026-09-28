# P/C differential adaptation necessity — seed 0

Frozen native backbone and final affine classifier; TRAIN-only source geometry/selection; OUTER_DEVELOPMENT evaluation only; formal final-heldout EEG reads: 0.
The historical strict freeze-P/C-only route failed. This experiment asks whether TRAIN-selected unequal P/C adaptation strengths transfer beyond the best homogeneous and update-norm-matched alternatives; it does not claim P invariance or C nuisance.

## Q1 — generic recalibration

Primary mean full-generic BA 0.742250 versus NoAdapt 0.737000. Differential adaptation harms 0.225 of held subjects by >0.01 BA and improves 0.450 by >0.01 BA; mean worst-quartile delta BA -0.012000. All five fold and subject effects are in OUTER_DEPLOYABLE_RESULTS.csv and SUBJECT_LEVEL_EFFECTS.csv.

## Q2 — off-diagonal response

TRAIN-selected alpha gap >=0.25 in 3/5 primary folds; gate A=False. The full 25-point TRAIN surfaces for every real partition are in ADAPTATION_RESPONSE_SURFACE.csv.

## Q3 — differential versus best homogeneous

Primary pooled paired BA delta 0.000750, 95% subject bootstrap [-0.001750, 0.003250]; gate B=False.

## Q4 — norm-matched homogeneous

Primary pooled paired BA delta 0.002750, 95% subject bootstrap [-0.000250, 0.006000]; gate C=False.

## Q5 — P/C tolerance

Mean held subject BA sweep range: P 0.018250, C 0.007500; these are descriptive tolerance differences, not a standalone claim of transfer. PC_ADAPTATION_TOLERANCE.csv gives every five-point sweep with BA/F1/NLL, representation drift and classifier-margin drift.

## Q6 — rollback

Primary full-rollback recovery relative to full generic across all subjects: P -0.004750 BA, C 0.003750 BA. Among 11 generic-harmed subject-folds (FullGeneric minus NoAdapt < -0.01 BA), mean P/C rollback recovery is 0.010909/0.008182 BA. These evaluation-only curves never select the deployed rates.

## Q7 — classifier actionability

Primary mean frozen-head P/C sensitivity fractions 0.474976/0.525024. PC_DECISION_LEVERAGE.csv also gives class-margin operator norms and realized P/C logit deltas; information is not equated with actionability.

## Q8 — partition specificity

Random p95 criterion=False; PCA criterion=False; supervised-decision honesty criterion=False. Primary mean Protected-minus-PCA and Protected-minus-supervised BA are 0.001000 and -0.000250. Each family received the same alpha/context/operator/pseudo-target protocol; all 100 random selected controls per fold are retained in RANDOM_PARTITION_DISTRIBUTION.csv.

## Q9 — replication

EEGNet/OpenBMI_ERP: positive folds 1/5; pooled BA delta -0.001242; consistent=False.
EEGConformer/OpenBMI_MI: positive folds 1/5; pooled BA delta 0.000250; consistent=False.

## Q10 — proceed?

Interpretation: `UNIFORM_SHRINKAGE_EXPLAINS_THE_GAIN`. Exact next action: `STOP_PC_SPECIFIC_ADAPTATION`. No new module was trained. Evaluation-only per-subject oracle results are nondeployable and not used to choose alphas.

## Final 15-fold completion table

| backbone | task | fold | rank_P | rank_C | context_budget | NoAdapt_BA | FullGeneric_BA | alpha_P | alpha_C | alpha_gap | Differential_BA | BestUniform_alpha | BestUniform_BA | Delta_vs_uniform_BA | NormMatchedUniform_BA | StrictProtect_BA | POnly_BA | P_rollback_recovery | C_rollback_recovery | P_drift | C_drift | P_classifier_leverage | C_classifier_leverage | PCA_differential_BA | SupervisedDecision_differential_BA | Random_differential_p95 | negative_transfer_any | primary_fold_result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EEGNet | OpenBMI_MI | 0 | 4 | 60 | 8 | 0.785000 | 0.785000 | 1.000000 | 1.000000 | 0.000000 | 0.785000 | 1.000000 | 0.785000 | 0.000000 | 0.785000 | 0.786250 | 0.785000 | 0.001250 | 0.000000 | 0.380715 | 0.860814 | 0.451270 | 0.548730 | 0.785000 | 0.785000 | 0.785000 | True | NOT_SUPPORTED |
| EEGNet | OpenBMI_MI | 1 | 3 | 61 | 8 | 0.775000 | 0.790000 | 0.750000 | 0.250000 | 0.500000 | 0.790000 | 0.250000 | 0.782500 | 0.007500 | 0.786250 | 0.778750 | 0.791250 | -0.011250 | 0.001250 | 0.240650 | 0.211789 | 0.436461 | 0.563539 | 0.782500 | 0.790000 | 0.790000 | False | DIFFERENTIAL_WINS |
| EEGNet | OpenBMI_MI | 2 | 4 | 60 | 16 | 0.688750 | 0.710000 | 1.000000 | 1.000000 | 0.000000 | 0.710000 | 1.000000 | 0.710000 | 0.000000 | 0.710000 | 0.688750 | 0.708750 | -0.021250 | -0.001250 | 0.658182 | 1.238940 | 0.493620 | 0.506380 | 0.710000 | 0.710000 | 0.711250 | False | NOT_SUPPORTED |
| EEGNet | OpenBMI_MI | 3 | 8 | 56 | 16 | 0.720000 | 0.710000 | 0.250000 | 1.000000 | 0.750000 | 0.725000 | 0.500000 | 0.725000 | -0.000000 | 0.711250 | 0.720000 | 0.722500 | 0.010000 | 0.012500 | 0.465893 | 2.950663 | 0.442409 | 0.557591 | 0.723750 | 0.722500 | 0.728750 | True | NOT_SUPPORTED |
| EEGNet | OpenBMI_MI | 4 | 5 | 59 | 64 | 0.716250 | 0.716250 | 1.000000 | 0.500000 | 0.500000 | 0.717500 | 0.750000 | 0.721250 | -0.003750 | 0.721250 | 0.713750 | 0.722500 | -0.002500 | 0.006250 | 0.516989 | 0.444337 | 0.551121 | 0.448879 | 0.721250 | 0.721250 | 0.721250 | True | NOT_SUPPORTED |
| EEGNet | OpenBMI_ERP | 0 | 6 | 58 | 64 | 0.820114 | 0.823788 | 0.000000 | 1.000000 | 1.000000 | 0.817538 | 0.250000 | 0.821061 | -0.003523 | 0.824091 | 0.817538 | 0.825909 | -0.006250 | 0.002121 | 0.000000 | 0.484164 | 0.699765 | 0.300235 | 0.821061 | 0.821061 | 0.821364 | False | NOT_SUPPORTED |
| EEGNet | OpenBMI_ERP | 1 | 9 | 55 | 8 | 0.893636 | 0.894053 | 0.000000 | 0.500000 | 0.500000 | 0.893939 | 0.250000 | 0.894356 | -0.000417 | 0.894508 | 0.893788 | 0.894242 | -0.000265 | 0.000189 | 0.000000 | 0.289644 | 0.606693 | 0.393307 | 0.893939 | 0.893674 | 0.894625 | False | NOT_SUPPORTED |
| EEGNet | OpenBMI_ERP | 2 | 8 | 56 | 64 | 0.780265 | 0.778068 | 0.250000 | 0.750000 | 0.500000 | 0.780000 | 0.500000 | 0.779811 | 0.000189 | 0.778826 | 0.778674 | 0.778902 | 0.000606 | 0.000833 | 0.094847 | 0.317743 | 0.639991 | 0.360009 | 0.779811 | 0.779811 | 0.779926 | False | NOT_SUPPORTED |
| EEGNet | OpenBMI_ERP | 3 | 4 | 60 | 8 | 0.884394 | 0.882462 | 0.000000 | 0.000000 | 0.000000 | 0.884394 | 0.000000 | 0.884394 | 0.000000 | 0.884394 | 0.881970 | 0.883939 | -0.000492 | 0.001477 | 0.000000 | 0.000000 | 0.506361 | 0.493639 | 0.884394 | 0.884394 | 0.884623 | False | NOT_SUPPORTED |
| EEGNet | OpenBMI_ERP | 4 | 5 | 59 | 8 | 0.835000 | 0.835076 | 1.000000 | 0.000000 | 1.000000 | 0.832538 | 0.000000 | 0.835000 | -0.002462 | 0.835189 | 0.837386 | 0.832538 | 0.002311 | -0.002538 | 0.311536 | 0.000000 | 0.406570 | 0.593430 | 0.835000 | 0.835000 | 0.835610 | True | NOT_SUPPORTED |
| EEGConformer | OpenBMI_MI | 0 | 7 | 25 | 8 | 0.825000 | 0.816250 | 0.750000 | 0.750000 | 0.000000 | 0.816250 | 0.750000 | 0.816250 | 0.000000 | 0.816250 | 0.823750 | 0.818750 | 0.007500 | 0.002500 | 0.528902 | 0.374067 | 0.401096 | 0.598904 | 0.817500 | 0.816250 | 0.819062 | True | NOT_SUPPORTED |
| EEGConformer | OpenBMI_MI | 1 | 5 | 27 | 32 | 0.762500 | 0.752500 | 0.750000 | 1.000000 | 0.250000 | 0.756250 | 1.000000 | 0.752500 | 0.003750 | 0.753750 | 0.766250 | 0.748750 | 0.013750 | -0.003750 | 1.152772 | 1.097742 | 0.355763 | 0.644237 | 0.752500 | 0.756250 | 0.757500 | True | NOT_SUPPORTED |
| EEGConformer | OpenBMI_MI | 2 | 5 | 27 | 64 | 0.642500 | 0.637500 | 0.750000 | 0.250000 | 0.500000 | 0.637500 | 1.000000 | 0.637500 | 0.000000 | 0.645000 | 0.641250 | 0.645000 | 0.003750 | 0.007500 | 1.017773 | 0.220956 | 0.444720 | 0.555280 | 0.640000 | 0.643750 | 0.642500 | True | NOT_SUPPORTED |
| EEGConformer | OpenBMI_MI | 3 | 8 | 24 | 16 | 0.720000 | 0.733750 | 1.000000 | 0.000000 | 1.000000 | 0.731250 | 1.000000 | 0.733750 | -0.002500 | 0.730000 | 0.726250 | 0.731250 | -0.007500 | -0.002500 | 10.313862 | 0.000000 | 0.401537 | 0.598463 | 0.732500 | 0.731250 | 0.735000 | True | NOT_SUPPORTED |
| EEGConformer | OpenBMI_MI | 4 | 8 | 24 | 64 | 0.732500 | 0.743750 | 0.500000 | 1.000000 | 0.500000 | 0.737500 | 0.500000 | 0.737500 | 0.000000 | 0.738750 | 0.733750 | 0.743750 | -0.010000 | 0.000000 | 0.300286 | 0.233399 | 0.503973 | 0.496027 | 0.737500 | 0.737500 | 0.738750 | True | NOT_SUPPORTED |

## Three cell-level summaries

- EEGNet/OpenBMI_MI: mean differential BA 0.745500, best-uniform BA 0.744750, paired delta 0.000750 [95% CI -0.001750, 0.003250], selected off-diagonal 3/5.
- EEGNet/OpenBMI_ERP: mean differential BA 0.841682, best-uniform BA 0.842924, paired delta -0.001242 [95% CI -0.002417, -0.000220], selected off-diagonal 4/5.
- EEGConformer/OpenBMI_MI: mean differential BA 0.735750, best-uniform BA 0.735500, paired delta 0.000250 [95% CI -0.002750, 0.002750], selected off-diagonal 4/5.

Primary interpretation: `UNIFORM_SHRINKAGE_EXPLAINS_THE_GAIN`.
Exact next action: `STOP_PC_SPECIFIC_ADAPTATION`.
Formal final-heldout EEG reads: `0`.
Branch: `codex/persist-eeg-pc-differential-adaptation-necessity-seed0-v1`.
Final commit SHA: recorded in the GitHub delivery response (a report cannot contain its own commit hash).
