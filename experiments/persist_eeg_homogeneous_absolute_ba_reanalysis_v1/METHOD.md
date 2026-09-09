# Method: absolute BA reanalysis

This is a statistics-only reanalysis of frozen source file
experiments/persist_eeg_matched_homogeneous_ensemble_control_v1/subject_level_results.csv
with SHA-256 a803c2683de8d44f5a08235186e9ba33129a3dc9f91f5ef3aacde9f435e51a7d. No EEGNet or LiteBN model was trained, loaded for
new inference, calibrated, or selected.

## Locked inputs

The four retained blocks are OpenBMI MI, OpenBMI ERP, OpenBMI SSVEP, and WBCIC
MI. The source ledger contains five historical folds, seeds {0,1,2}, and all
unordered pairs (0,1), (0,2), (1,2). The primary rule is LOGIT50.

EE is the fixed 50/50 fusion of two EEGNet seeds. LL is the fixed 50/50 fusion
of two LiteBN seeds. EL retains both fixed orientations, EL_A and EL_B, and
uses their arithmetic outcome mean. It is not a four-model ensemble.

## Aggregation and uncertainty

For every block, subject, and primary family, the 15 fixed fold by seed-pair BA
values are first averaged. EL has both orientations equally represented inside
each fixed replicate. The family BA is then the ordinary mean of those
subject-level values across matched held-out subjects. The subject is therefore
the final statistical unit; trials are never pooled and folds/pairs are never
treated as independent observations.

BA and contrasts are percentage points. Family CIs use 10,000 subject
bootstraps. EL minus EE, EL minus LL, and EL minus 0.5 times (EE plus LL) use
paired subject bootstrap. Global equal-block CIs resample subjects independently
within every block and then average the four block means. A separately labeled
dataset-balanced result averages the three OpenBMI task blocks before averaging
OpenBMI and WBCIC 50/50.

No best subject-specific constituent, seed pair, fold, orientation, fusion
weight, calibration, routing, or gating is used. The former post-hoc
gain-over-best-constituent quantity is excluded from the primary table.
