# PERSIST-EEG Cumulative Subject Decision Drift Audit V1

Seed-0 EEGNet only; OpenBMI/WBCIC canonical folds 0--4. Each fold has five deterministic fixed-sentinel meta-fold continuations, each exactly one A-only natural epoch from the canonical checkpoint. Sentinel subjects are excluded from every A batch. K=4 class-balanced BBR/CE certificates are evaluated on source/refit trials and compared with held-out remaining source/refit B_out.

This is a signal audit only: no guard, rollback, correction, new objective, hyperparameter search, seed 1/2, second backbone, WBCIC outer-10, or OpenBMI sealed cohort. Runtime, checkpoints, cache and raw EEG stay outside the committed artifact set.
