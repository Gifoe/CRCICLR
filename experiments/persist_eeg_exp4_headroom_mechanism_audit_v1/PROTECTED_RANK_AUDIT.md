# Protected rank audit

Old characterization: **CAP_SATURATION_AFTER_OLD_WEAK_P_U_FLIP_QUALIFICATION**.

The old code did apply P, signed-U, and nonzero flip conditions before taking the top eight; it did not literally select arbitrary directions. However, the flip threshold was only `>0`, every run had at least eight such candidates, and all runs therefore saturated the cap. The repaired bank requires source-only P+U plus both exact finite-D and Jacobian ratios above matched-random controls. Per-fold/seed counts are in `results/PROTECTED_DIRECTION_AUDIT.csv` and `results/PROTECTED_RANK_SUMMARY.csv`.
