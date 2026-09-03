# Autonomous decision

terminal = SSPG_MECHANISM_SUPPORTED_PERFORMANCE_INSUFFICIENT

SSPG reduces local independent B_out harm on both datasets, but its matched unseen-subject BA is negative on both datasets. Therefore this seed-0 result does not justify three-seed confirmation under the frozen strong-signal gate.

`THREE_SEED_CONFIRMATION_SCIENTIFICALLY_JUSTIFIED = NO`
`AUTO_RUN_SEED1_SEED2 = NO`
`seed1_run = false`; `seed2_run = false`; `second_backbone_run = false`; `WBCIC_outer_opened = false`; `OpenBMI_sealed_opened = false`.

|dataset|TaskOnly BA|SSPG BA|delta pp|95% CI pp|nonnegative folds|
|---|---|---|---|---|---|
|OpenBMI|0.819074|0.818519|-0.056|[-0.426, +0.241]|3/5|
|WBCIC|0.791415|0.790684|-0.073|[-0.207, +0.049]|2/5|
