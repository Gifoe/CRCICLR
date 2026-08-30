# Repair-R1 rationale

## Hypothesis

The V3 Bures operator reaches target-subject affinity but transports
class-discriminative structure as well.  A source-only task-protected
projection should improve class fidelity while retaining non-trivial target
affinity.

## Exact scientific change

For every source anchor and Bures subject displacement `delta_subject`,
estimate one protected direction from the source-training class-centroid
difference `d = mean(class=1) - mean(class=0)`.  The repair uses

`delta_R1 = delta_subject - d (d^T delta_subject)`

with `d` normalized.  The clean feature representation is unchanged.  The
matched random control receives a signed displacement with the same
Euclidean and pooled-whitened norm as `delta_R1`.  The candidate validity gates
and source split remain unchanged.

## What this repairs

It directly addresses V3's low class fidelity/coverage and the V2 finding that
structured transport did not beat matched random.  It does not change the
effect-size target, folds, seeds, data sessions, or final architecture target.

## Frozen R1 recipe budget

The repaired primary has exactly six source recipes:
`q in {0.25, 0.50}` crossed with `lambda_T in {0.25, 0.50, 1.00}`.
The alpha ladder remains V3's `{0.25, 0.50, 0.75, 1.00}` and is not selected
from outcomes.  Geometry is computed once after the fixed warm-up and remains
detached while the final block and head train.  ERM, Mixup, V3 Bures and the
matched R1 random control are comparators, not additional primary recipes.
