# AMSE-v1 aggregation erratum

The original V1 reporting used `mean_subject(max(EEGNet_subject_BA,
LiteBN_subject_BA))` as `best_single_ref_BA`.  That is a subject-wise oracle,
not a realizable fixed single model.  The corrected primary definition is
`max(mean_subject(EEGNet_BA), mean_subject(LiteBN_BA))`, with the selected
dataset-level model held fixed for every subject and bootstrap draw.

This is a reporting-only correction.  V1 numerical model outputs and
checkpoints are unchanged, no V1 rerun is required, and the V1 terminal remains
`AMSE_SEED0_FAIL`.  The subject-wise oracle is retained only as a descriptive
diagnostic in the V2 report.
