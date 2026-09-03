# PERSIST-SSPG seed-0 final report

terminal = SSPG_MECHANISM_SUPPORTED_PERFORMANCE_INSUFFICIENT

Primary comparison is SSPG vs TASK_ONLY_MATCHED; ANCHOR is reference only. CIs are 10,000-draw paired biological-subject bootstrap intervals.

|dataset|TaskOnly BA|SSPG BA|SSPG-TaskOnly pp|95% CI pp|SSPG-Cross pp|SSPG-Random pp|nonnegative folds|
|---|---|---|---|---|---|---|---|
|OpenBMI|0.819074|0.818519|-0.056|[-0.426, +0.241]|-0.037|-0.056|3/5|
|WBCIC|0.791415|0.790684|-0.073|[-0.207, +0.049]|-0.110|-0.085|2/5|

Independent B_out harm:

|dataset|mean positive harm Task|mean positive harm SSPG|reduction|CI lower|frequency Task|frequency SSPG|
|---|---|---|---|---|---|---|
|OpenBMI|0.00016198499|8.6312182e-05|7.5672812e-05|4.0923516e-05|0.4528|0.3567|
|WBCIC|8.0511322e-05|6.0059832e-05|2.045149e-05|6.280241e-06|0.4517|0.4344|

Judgment: the mechanism endpoint improves (harm reduction in both datasets), but downstream classification does not improve and controls are not beaten. This is not a strong signal and no seed1/seed2 confirmation was run.

SSPG_SEED0_STRONG_SIGNAL = NO
THREE_SEED_CONFIRMATION_SCIENTIFICALLY_JUSTIFIED = NO
AUTO_RUN_SEED1_SEED2 = NO
seed1_run = false
seed2_run = false
second_backbone_run = false
WBCIC_outer_opened = false
OpenBMI_sealed_opened = false
