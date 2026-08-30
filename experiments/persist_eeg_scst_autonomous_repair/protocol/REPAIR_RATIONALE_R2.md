# Repair-R2 rationale: low-rank local target-conditional OT

R1 reached target-subject affinity and meaningful displacement, but its global
task-protected Bures operator remained class-unsafe (class fidelity about
0.31--0.33, coverage about 0.40--0.41) and failed every utility comparison.
The signature is consistent with a global Gaussian model mismatch rather than
weak transport.  R2 therefore tests one hypothesis only:

> A source-only local, low-rank target-conditional transport map can retain
> target affinity while avoiding the heterogeneous global directions that
> caused R1's class-fidelity and coverage failure.

For each source anchor and target subject, R2 selects the nearest 12 same-class
rows from the opposite cross-fit half for both source and target.  It forms
local shrinkage covariances, computes a PCA basis on the pooled local residuals,
retains the smallest rank explaining at least 90% of local variance (capped at
rank 8), applies the regularized Bures map only in that basis, and leaves the
orthogonal residual unchanged.  A local map with fewer than two rows on either
side yields no valid candidate.  The 90% threshold, neighbourhood size, rank
cap, and all source gates are fixed before any R2 result is read.

R2 keeps the same six `(q, lambda_T)` recipes, alpha ladder, folds, seeds,
bootstrap, source split, matched random construction, ERM, Mixup, and R1
comparator.  No future, outer, sealed, or S3 resource is opened.
