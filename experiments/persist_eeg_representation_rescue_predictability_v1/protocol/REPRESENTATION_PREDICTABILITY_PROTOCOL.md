# Representation predictability protocol

This is a development-outer-only diagnostic replay of frozen B0, X, and XS checkpoints. It inherits the exact repaired ERP numerical semantics, B0 definitions, folds, normalizers, source/evaluation sessions, and aligned trial identities from `persist_eeg_stable_rescue_predictability_audit_v1`.

Only B0/candidate disagreements enter diagnostic fitting. The targets are `CLEAN_RESCUE_VS_HARM` and `SAFE_SWITCH`. F0 is the exact 13-feature confidence control. F1 appends ordered, unsorted centered logits, probabilities, their differences, and the exact class-1-minus-class-0 signed margins for binary tasks. F2 appends separate full 64-dimensional B0 and 128-dimensional candidate pre-head embeddings. No label-derived feature, identifier, PCA, learned projection, feature selection, or hyperparameter search is allowed.

Every task is fit separately with `StandardScaler` followed by `LogisticRegression(penalty="l2", C=1, solver="lbfgs", max_iter=5000, class_weight="balanced", random_state=0)`. Cross-fitting uses the exact five outer folds. XS training pools seeds 0/1/2 without a seed feature and assigns equal total fit mass to each real subject across replicas. The same fold-specific model is evaluated separately on each seed. Leave-one-seed-out analysis excludes both the held fold and held seed.

Policy thresholds 0.05 through 0.95 in steps of 0.01 are selected only on the four training folds by subject-equal BA; ties choose the larger threshold. Policy uncertainty uses 10,000 paired real-subject bootstrap resamples, keeping XS replicas together.

F0 must reproduce the prior SAFE AUC and policy delta within 1e-8. Failure terminates interpretation. No EEG network is trained, and internal-heldout, final-heldout, final-test, and sealed-test artifacts are forbidden.
