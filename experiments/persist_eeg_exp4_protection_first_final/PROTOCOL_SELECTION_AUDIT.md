# Protocol selection audit

The primary deployment question was kept as `S1 anchor -> S2 global representation update -> unseen-subject S3 evaluation`. This is the closest implementation of the requested past-session-to-future-session setting and uses the existing frozen five-fold WBCIC development scope.

For each fold, the outcome subjects are never used for anchor fitting, persistence-basis construction, adapter fitting, or generic selection. The discovery/decision subjects are the next frozen fold, and the remaining three folds are model-fit subjects. The S3 labels of the outcome subjects are read only by `compute_dev` as the subject-level endpoint.

The protected basis is the first rank-four block (`P01_04`) of the fold-specific S1/S2 cross-session subject-centroid persistence basis. The assignment is inherited from the frozen WBCIC actionability audit (`protected_utility_gate=true`); it is not reselected from S3. `P05_08` is the rank-matched persistence-only control. PCA, identity, and random controls are constructed at the same rank.

The anchor is EEGNet trained only on S1 with a frozen linear head. The adapter is a zero-initialized linear residual with the same parameter count, optimizer, training data, and epoch budget for Generic and all guards. V1 uses the structural equation

`h_guard = h_anchor + (I - U_P U_P^T) A(h_anchor)`.

V2 was introduced only after V1 showed that coordinate preservation did not preserve Exp3 decision response. It added a minimum-norm frozen-head response correction; it was rejected because it caused a large negative performance shift. V3 used a fixed `alpha=0.25` partial correction to test the specific over-constraint diagnosis; it also failed the gate. These changes are recorded in `ITERATION_LEDGER.md`, not silently folded into V1.

The development gate requires generic competence, a practical paired Guard−Generic improvement, lower negative transfer or lower-tail improvement, protected-mechanism evidence, and specificity over matched controls. None of the three variants met all conditions. Therefore the final protocol file is explicitly non-authorizing, and outer evaluation is forbidden.
