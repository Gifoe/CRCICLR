# EEGNet Core–Residual premise validation

Frozen canonical EEGNet / OpenBMI_MI / seed 0 / five folds, 64-D embedding. TRAIN_GEOMETRY alone fitted all subspaces, decoders and grouped-OOF stackers. OUTER_DEVELOPMENT was evaluation-only. CHECKPOINT_VALIDATION is separately selection-exposed. Formal final-heldout EEG reads: **0**.

The source `U_NOT_P` rank is zero in all five folds and is retained as `NOT_ESTIMABLE` in the construction audit; no metric is fabricated for it.

## Q1–Q2. Geometry versus prediction

G held cross-subject relation **0.922**, PCA_R **0.902**, supervised-decision_R **0.963**. Shared-relation alignment: G **0.922**, PCA_R **0.901**, supervised-decision_R **0.963**. G subject-task heterogeneity **0.044** and session-task instability **0.144**. The locked reproducible-geometry gate is **not met**; relation +0.03 fold counts versus PCA/supervised are **{'PCA_R': 1, 'SUPERVISED_DECISION_R': 0}**. Do not infer an advantage merely from the mean.

Frozen held cross-subject BA: G **0.716**, PCA_R **0.733**, supervised-decision_R **0.734**. The predictive-maximality contrast is **supported**. FISHER_1D is explicitly rank one, not a matched competitor. Paired 20,000-draw biological-subject intervals are in the G-versus-PCA and G-versus-supervised CSVs.

## Q3–Q4. Residual location and error rescue

Residual-only BA (P_NOT_U / NEITHER / C_CURRENT): **0.708 / 0.700 / 0.611**. Subject-equal rescue fractions P(residual correct | G wrong): **0.292 / 0.334 / 0.465**. Error transitions are trial-level in `RESIDUAL_ERROR_COMPLEMENTARITY.csv`. `ORACLE_UNION` is a **NONDEPLOYABLE_COMPLEMENTARITY_UPPER_BOUND**; it is not an ensemble or achievable held predictor. Descriptive residual-role labels: **MISSING_UTILITY_IN_NONPERSISTENT_RESIDUAL**.

## Q5–Q6. Deployable joint, capacity, and stacked tests

G-only BA **0.716**. G+P_NOT_U **0.718**, G+NEITHER **0.717**, and G+C_CURRENT **0.668**. TRAIN-PCA dimension-matched BA for the corresponding residuals: **0.710 / 0.718 / 0.645**. G+C_CURRENT spans the full embedding; its D>64 feature-count comparator is TRAIN PCA64 zero-padded to D, with effective rank 64 disclosed rather than falsely called a novel higher-rank representation. The two-logit stacker used five biological-subject-grouped OOF TRAIN folds. Its BA for P_NOT_U / NEITHER / C_CURRENT: **0.721 / 0.727 / 0.723**. Residual-specific deployable/capacity/stacked gates: **{"C_CURRENT": {"RESIDUAL_COMPLEMENTS_G": false, "capacity_BA_advantage_folds": 5, "capacity_NLL_advantage_folds": 1, "capacity_specific": true, "joint_BA_gain_at_least_0_005_folds": 0, "joint_NLL_gain_folds": 0, "rescue_fraction_equal_fold_subject_equal": 0.4647020386270553, "stack_favorable": true, "stacked_BA_equal_fold": 0.7228515625, "stacked_NLL_equal_fold": 0.5639005835982384}, "NEITHER": {"RESIDUAL_COMPLEMENTS_G": true, "capacity_BA_advantage_folds": 3, "capacity_NLL_advantage_folds": 0, "capacity_specific": true, "joint_BA_gain_at_least_0_005_folds": 3, "joint_NLL_gain_folds": 0, "rescue_fraction_equal_fold_subject_equal": 0.3343340528231934, "stack_favorable": true, "stacked_BA_equal_fold": 0.72734375, "stacked_NLL_equal_fold": 0.5594129928544563}, "P_NOT_U": {"RESIDUAL_COMPLEMENTS_G": false, "capacity_BA_advantage_folds": 3, "capacity_NLL_advantage_folds": 0, "capacity_specific": true, "joint_BA_gain_at_least_0_005_folds": 2, "joint_NLL_gain_folds": 0, "rescue_fraction_equal_fold_subject_equal": 0.29218934738185226, "stack_favorable": true, "stacked_BA_equal_fold": 0.7208984375, "stacked_NLL_equal_fold": 0.5601213263102869}}**. A trial-level oracle rescue fraction alone never supports a model premise.

## Q7–Q8. Mechanism and next action

Core–Residual premise: **NOT_ESTABLISHED**. Primary interpretation: **`G_NOT_DISTINCT_FROM_GENERIC_PREDICTIVE_SUBSPACE`**. Exactly one next action: **`STOP_P_SPECIFIC_MODEL_DESIGN`**. No neural architecture was implemented or trained here.

Case C's generic-subspace sufficiency check asks whether the *same* alternative has at least G's relation cosine and BA in every fold: **{'PCA_R': False, 'SUPERVISED_DECISION_R': True}**. The first aggregation incorrectly returned Case E despite this pattern. The Case C mapping was corrected after aggregate inspection; no representation, decoder, fold outcome or frozen scientific threshold changed. Both aggregate versions are retained on the server.

## Integrity boundary

All five G geometries and pre-existing family held metrics were reproduced against the immutable source, with fold-level hashes in the construction and evaluation audits. All PCA/supervised/Fisher bases were fitted before any held-role read. Normalizer, checkpoint, parameters and BatchNorm state were checked. Source-specificity results already inspected OUTER_DEVELOPMENT historically; this is a frozen follow-up and the OUTER evidence is descriptive, not an untouched confirmatory test. Formal final-heldout EEG reads: 0.
