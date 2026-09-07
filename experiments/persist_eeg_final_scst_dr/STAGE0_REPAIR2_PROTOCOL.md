# Stage-0 Repair 2 protocol

## Question

Can the frozen subject-class residual direction become a valid transport when
each independent-session candidate is limited by source-session same-class
support, without changing the direction or the original validity gates?

## Sole scientific change

The layer is fixed to `final_embedding`, the maximum step is fixed to 0.25, and
the residual direction is unchanged.  For each candidate, the operator chooses
the largest value in `{0, 1/64, ..., 16/64}` whose mean 3NN distance to real
same-class model-fit Session-1 centroids does not exceed the Session-1-only 95th
percentile leave-one-subject-out clean radius.  Session 2 is used only after the
operator is fixed, for the same independent subject-, class-, and manifold-
fidelity tests used previously.

The candidate support set contains all real bank-session subject-class
centroids.  This does not create a self-neighbor because every candidate query
is an independent-session centroid or trial.  Only construction of the clean
radius excludes the queried bank centroid itself.

Norm-matched random perturbations use exactly the realized constrained SCST L2
norm.  Other directional controls receive the same candidate-specific
`alpha_star`; same-class Mixup remains at 0.5.

## Hard terminal

All four retained settings must pass the unchanged competence, stability,
subject-fidelity, random-advantage, class-fidelity, 1.25 manifold-ratio, and
off-rate gates.  Any failure ends transport development permanently.  There is
no Repair 3, threshold relaxation, setting removal, future-performance access,
or sealed access.
