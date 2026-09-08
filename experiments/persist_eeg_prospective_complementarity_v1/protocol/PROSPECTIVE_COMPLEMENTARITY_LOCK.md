# Prospective complementarity protocol lock

Lock timestamp (UTC): `2026-09-08T19:05:03.388955+00:00`

Fixed analysis: no training, fusion fitting, alpha tuning, checkpoint
reselection, preprocessing changes, or final held-out-subject access. The
primary predictor is subject-balanced exclusive-correct complementarity on
INNER_VAL future-session trials (`C_subject`). The primary outcome is OUTER_DEV
`G_best` under the already locked raw-logit 50/50 fusion. Primary correlation
is Spearman with block-aware resampling by dataset x backbone. Partner ranking
uses sign(Delta_C)==sign(Delta_G), tie tolerance 1e-8.

Only units whose predictor was frozen before the matching OUTER_DEV outcome
existed can be confirmatory. Earlier outcomes remain labeled
`PREEXISTING_OUTCOME` and are supportive/post-hoc only. Fixed pairs are
EEGConformer, CBraMod, CodeBrain crossed with LiteBN and EEGNet; datasets are
OpenBMI and WBCIC; folds are 0-4. Final holdouts and WBCIC true outer subjects
are outside scope.
