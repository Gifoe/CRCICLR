# BNCI2015-001 P3 architecture replication

B0 Full is a frozen reuse. Only B1 SameScale-63, B2 ScaleCollapse, B3 SingleSpatialBasis, and B4 OneStageBackend are trained.
Every B1--B4 fold reuses the frozen full fold's cached [13,1000] tensors, S1-only inner-train normalizer, exact 60x20 authoritative episode manifest, split, AdamW lr=3e-4, weight decay=5e-4, ordinary CE, clip=5, AMP convention, 60 epochs and S2 inner-validation earliest-tie checkpoint rule.
Support is 64 S1 trials from four inner-train subjects; query is 64 S2 trials from four different inner-train subjects. Validation and outer subjects never enter an episode.
