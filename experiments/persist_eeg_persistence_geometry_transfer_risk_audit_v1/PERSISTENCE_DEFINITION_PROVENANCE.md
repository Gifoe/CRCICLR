# Persistence definition provenance

The implementation is a conservative MI-only extraction of Signed V3.1.
The train-only A representation is whitened to the active top-20 rank.  For
each binary class, subject/session centroids are centered across A subjects;
the symmetrized S1/S2 cross-covariance is eigendecomposed and ordered by
descending eigenvalue.  Deterministic max-size-four blocks and the 200-draw
subject-permutation null define `persistence_supported` exactly before any
discovery query outcome is read.

Block utility uses the Signed V3.1 ridge-probe convention (first 24 latent
coordinates), five deterministic subject-disjoint splits, 100 SHA256-seeded
matched-rank non-Protected interventions, and the same positive lower-CI
Protected rule for absolute and specificity utility.  The current fold's
A-only embeddings are always the basis; historical spectra are never loaded.

WBCIC's historical direction implementation is algebraically compatible with
the symmetrized subject-centroid covariance above.  Rather than mixing its
historical coordinates, this audit applies the same unified definition to
WBCIC S1/S2, which is the conservative choice for a cross-dataset claim.

The only primary descriptor is

`align = 0.5*(cos(v_B,S2, v_A,S1) + cos(v_B,S1, v_A,S2))`, `novelty = 1-align`.

No metric, block, cap, query rule or threshold is selected from transfer
outcomes.
