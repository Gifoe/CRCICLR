# Frozen protocol

- EEGNet; OpenBMI and WBCIC; folds 0--4; seed 0 only.
- K=4, m_per_class=16, blocks 1--4 certificate and block 5 independent B_out.
- AdamW lr=3e-5, weight_decay=5e-4, gradient clip=5, two continuation epochs, full trainable parameter scope.
- kappa=0.20; BN running statistics frozen; Adam moments receive A/task gradients only.
- SSPG uses R(Delta)=mean ReLU(gbar_s dot Delta)^2, bounded projection and frozen backtracking multipliers.
- Outcome is biological-subject paired Balanced Accuracy, opened only after the committed pre-outcome lock.
- No seed 1/2, second backbone, WBCIC outer-10, OpenBMI sealed/confirmation cohort, K search, or outcome-based tuning.

Machine-readable lock: `results/PRE_OUTCOME_LOCK.json`.
