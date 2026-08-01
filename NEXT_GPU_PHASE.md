# Next GPU phase (not executed)

The CPU phase did not download CBraMod, call CUDA, extract real embeddings, train a real task head, or run real TTA.

Required order for the later GPU phase:

1. Obtain and checksum the public CBraMod checkpoint; record license and exact revision.
2. Load only the subject roles defined in `data/splits`; reserve subject-level early stopping inside `task_head_train`.
3. Extract frozen-backbone representations from cached windows. Normalization may be per-window or fitted only on U_s/training data; V_s statistics must never transform U_s.
4. Train the HMC and EEGMMIDB task heads only on their task-head roles. CAP inherits the HMC head.
5. Fit meta-risk predictors only on `meta_risk_train` subjects. CAP inherits the HMC predictor.
6. Produce the three parquet interfaces shown below. Context and adaptation features may use only U_s. Future risk and task metrics may use V_s only during offline evaluation.
7. Fit one simultaneous residual quantile from calibration subjects and freeze it before final-test decisions.
8. Run schema, grid-completeness, subject-role, context/future, and provenance checks before computing final metrics.

| file | rows | bytes |
| --- | --- | --- |
| subject_context_features.parquet | 30 | 18516 |
| subject_action_surface.parquet | 1800 | 84948 |
| subject_decisions.parquet | 30 | 11935 |

Pooled float32 embeddings for 250,101 windows require about 0.19 GiB at dimension 200 or 0.48 GiB at dimension 512, before metadata. Dense token embeddings could require 5–20 GiB depending on token count. Current free space (283.28 GiB) is sufficient for these scenarios but must be rechecked before execution.
