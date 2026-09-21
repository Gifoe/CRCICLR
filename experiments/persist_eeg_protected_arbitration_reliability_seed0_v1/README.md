# PERSIST EEG protected arbitration reliability — seed 0 v1

This experiment evaluates frozen EEGNet and EEGConformer checkpoints on OpenBMI MI and SSVEP outer-development data. It reuses the prior protected pathway artifacts verbatim. It does not retrain a backbone or native head, reselect protected dimensions, or access final-heldout subjects.

Each policy is fit on TRAIN trials only. Regularization and abstention thresholds are chosen from leave-subject-out TRAIN predictions; the selected estimator is then refit on all TRAIN trials and evaluated exactly once on the frozen outer-development session. Failed integrity checks are written as `FAIL_CLOSED` cells and excluded from aggregate claims.
