# Control audit

Primary comparator: TASK_ONLY_MATCHED. CROSS_SUBJECT_K4 tests whether K4 averaging without same-biological-subject coherence explains the result. RANDOM_DIRECTION tests an arbitrary equal-norm perturbation under the same trigger regime.

|dataset|method|BA|delta vs TaskOnly (pp)|
|---|---|---|---|
|OpenBMI|TASK_ONLY_MATCHED|0.819074|+0.000|
|OpenBMI|SSPG|0.818519|-0.056|
|OpenBMI|CROSS_SUBJECT_K4_GUARD|0.818889|-0.019|
|OpenBMI|RANDOM_DIRECTION_GUARD|0.819074|+0.000|
|WBCIC|TASK_ONLY_MATCHED|0.791415|+0.000|
|WBCIC|SSPG|0.790684|-0.073|
|WBCIC|CROSS_SUBJECT_K4_GUARD|0.791781|+0.037|
|WBCIC|RANDOM_DIRECTION_GUARD|0.791537|+0.012|
