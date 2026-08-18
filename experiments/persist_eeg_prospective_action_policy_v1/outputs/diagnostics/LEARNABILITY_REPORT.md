# Actionability learnability

All metrics are grouped out-of-sample; no random row split is used.

- `openbmi_dda_block` / `leave_one_outer_fold_out`: best `P+U+D`, R2=0.7809, RMSE=0.0212, Spearman=0.8510.
- `openbmi_dda_block` / `leave_one_run_out`: best `P+U+D`, R2=0.7563, RMSE=0.0224, Spearman=0.8534.
- `openbmi_sample_router` / `leave_one_subject_group_out`: best `all_legal`, R2=0.2342, RMSE=0.3641, Spearman=0.3184.
- `wbcic_development_block` / `leave_one_backbone_out`: best `D`, R2=0.5234, RMSE=0.0316, Spearman=0.4775.
- `wbcic_development_block` / `leave_one_fold_out`: best `D`, R2=0.6663, RMSE=0.0264, Spearman=0.4713.

`U` is marked unavailable for the historical sample router because no utility estimate independent of the target trial outcome exists. Filling it with realised rescue/harm would be leakage.

Same-cell DDA/WBCIC signed utility is excluded. Only explicitly cross-fitted utility priors enter the corresponding legal feature set.

`OUTER_TEST_USED = false`.
