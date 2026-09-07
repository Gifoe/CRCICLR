# Alignment audit

The anchor is S1-only EEGNet. For each development fold, all non-outcome subjects' S1/S2 representations are used to compute a centered, ridge-whitened cross-session centroid covariance. The top eight orthonormal directions are the only candidate pool. The sealed outer split is never opened. See `results/ALIGNMENT_AUDIT.csv`.
