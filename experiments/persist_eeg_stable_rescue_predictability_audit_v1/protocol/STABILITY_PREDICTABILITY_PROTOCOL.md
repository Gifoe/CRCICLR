# Stability and predictability protocol

- Scope: development outer subjects only; OpenBMI MI, ERP, SSVEP, and WBCIC MI.
- Stage 0 audits the five previously failed ERP cells. ERP enters the full analysis only after exact source, checkpoint, normalizer, split, labels, session, trial count, and historical metrics are reproduced.
- ERP replay alternatives are restricted to historically plausible numerical execution semantics; no configuration is chosen for favorable BA.
- `B0_FIXED`: exact seed-0 baseline from the XS development experiment, reused identically for XS seeds 0/1/2 within each task/fold.
- Fitted diagnostic: `StandardScaler` plus L2 `LogisticRegression(C=1, solver=lbfgs, max_iter=5000, class_weight=balanced, random_state=0)`.
- Features: predeclared CONF_ONLY and MECH sets; no true/predicted class identity, subject, fold, or seed is a predictor.
- Cross-fitting: train on four outer folds and evaluate the fifth. The scaler, coefficients, and threshold use training folds only.
- XS primary model: seed-agnostic, pooling three seed replicas in training with subject-equal total sample weights.
- Threshold grid: 0.05 through 0.95 in 0.01 steps; ties select the larger threshold.
- Bootstrap: 10,000 real-subject resamples; XS replicas stay together.
- No EEG model training, neural router, feature tuning, C tuning, internal-heldout access, or final-test access.
- Both CONF_ONLY and MECH are always fit and reported. Predicted and true class identities remain metadata only and are never predictors.
- The ERP terminal is exactly one of `ERP_FULLY_INCLUDED`, `ERP_X_PARTIAL_XS_FULLY_INCLUDED`, `ERP_SEED0_ONLY`, `ERP_CROSSSEED_INCOMPLETE`, or `ERP_PROTOCOL_FAIL`.
