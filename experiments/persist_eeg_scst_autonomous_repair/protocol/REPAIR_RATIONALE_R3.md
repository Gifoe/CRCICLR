# Repair-R3 rationale: task-protected projection of local OT

R2 improved target-distance and target-NLL affinity but reduced class fidelity
and coverage further (0.196/0.224 and 0.337/0.360 for OpenBMI/WBCIC) and failed
all utility comparisons.  The remaining single, preregistered hypothesis is
that the local OT displacement still contains a task-semantic component.  R3
therefore applies a rank-one source-only projection to the already-frozen R2
displacement:

> `d_R3 = (I - P_task) d_local`, where `P_task` is the projector onto the
> normalized source-fit-half class-centroid difference.

The projection is computed separately within each source training unit and is
applied before the existing R2 candidate gates.  No neighborhood, PCA rank,
covariance regularization, recipe, alpha ladder, split, fold, seed, or gate is
changed.  ERM, Mixup, the historical V3 comparator, and matched random remain
the same controls.  Only OpenBMI session-1 to session-2 and WBCIC S1 to S2 are
used; future, outer, sealed, and S3 resources remain closed.  R3 is the final
constructive round; a failed source gate terminates the constructive search.
