# Canonical block protocol audit

| dataset   |   method_development_subjects |   retained_subjects |   retention_rate |   canonical_blocks |   max_based_rate | subjects_pass   | blocks_pass   | quantile_pass   | pass   |
|:----------|------------------------------:|--------------------:|-----------------:|-------------------:|-----------------:|:----------------|:--------------|:----------------|:-------|
| hmc       |                            90 |                  90 |                1 |               1392 |                0 | True            | True          | True            | True   |
| eegmmidb  |                            65 |                  65 |                1 |                390 |                0 | True            | True          | True            | True   |

Blocks are canonical physical structures and are not duplicated by source seed. HMC blocks are contiguous, non-overlapping, and never cross recordings. EEGMMIDB blocks preserve one original task run each.
