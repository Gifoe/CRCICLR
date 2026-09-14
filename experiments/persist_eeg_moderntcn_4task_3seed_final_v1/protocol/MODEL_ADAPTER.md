# Model adapter

The cache boundary is `B,C,T`. The official ModernTCN classification wrapper consumes `B,T,C`, so the adapter performs one axis transpose and invokes the unmodified vendored classifier. There is no channel reordering, filtering, montage feature, task-specific architecture change, or output calibration. Official RevIN remains enabled in addition to the mandatory project-level train-only z-score.

