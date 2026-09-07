# Engineering repair log

- Created an independent stress-test branch from `3366d419cfd5d26e88f5de3d751199d9068b649e`.
- Rejected the legacy OpenBMI baseline loader for this task because it reads the internal-holdout membership list during split construction. The replacement loader reads only the already authorized 40-subject cache and verifies exact pool equality.
- Selected the validated V7 OpenBMI CompactEEGConformer rather than silently substituting DeepConvNet.
- The initial fold-0/seed-0 engineering smoke used a single linear DANN adversary. Its high-strength source identity increased rather than decreased, so it was invalid as a manipulation implementation and was excluded before the formal grid. DANN was repaired to the standard fixed two-layer MLP domain classifier. The decision used only the source identity diagnostic; no lambda, task architecture, fold, seed, or outcome-selection rule changed.
