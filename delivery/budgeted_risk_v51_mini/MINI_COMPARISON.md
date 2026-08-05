# Mini S2/S3 comparison

| dataset | scheme | raw_spearman | raw_mae_improvement | q_mean | mean_seed_violation | worst_seed_violation | max_seed_cp_upper | relative_gain_vs_global | paired_gain_ci_low | oracle_gain_recovered | sentinel_delta | sentinel_transition_rate | q_driver_unstable_fold_rate | evaluation_loo_sign_stable | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eegmmidb | S2_EXACT_2TRAIN_2CAL | 0.6048 | 0.7874 | 4.2931 | 0.0831 | 0.1231 | 0.2111 | -0.0219 | -0.1448 | -0.1919 | 0.3815 | 0.3908 | 0.1600 | True | False |
| eegmmidb | S3_EXACT_2TRAIN_2CAL_SCALED | 0.6048 | 0.7874 | 4.7116 | 0.0431 | 0.0769 | 0.1550 | -0.0424 | -0.2194 | -0.3714 | 0.4492 | 0.4585 | 0.2000 | True | False |
| hmc | S2_EXACT_2TRAIN_2CAL | 0.5017 | 0.1191 | 4.2743 | 0.0956 | 0.1333 | 0.2071 | -0.0366 | -0.3018 | -0.1266 | 0.2400 | 0.2267 | 0.4000 | True | False |
| hmc | S3_EXACT_2TRAIN_2CAL_SCALED | 0.5017 | 0.1191 | 3.1692 | 0.0578 | 0.0778 | 0.1411 | -0.1116 | -0.5509 | -0.3857 | 0.3133 | 0.3000 | 0.5200 | True | False |
