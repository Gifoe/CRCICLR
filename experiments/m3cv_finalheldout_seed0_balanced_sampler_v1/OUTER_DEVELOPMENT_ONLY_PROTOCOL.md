# M3CV seed-0 hierarchical balanced-sampler variant

- The frozen 93-subject cohort, 75/18 development/final-heldout split, and five 48/12/15 development folds are unchanged.
- This run uses development data only: final-heldout metadata, signals, labels, checkpoint selection, and evaluation are excluded.
- The only changed component is the training sampler: subject -> session -> class -> trial, with equal train-subject/session/class/cell exposure in every epoch.
- Each cell uses deterministic cyclic shuffled trial permutations; a trial is never repeated inside one cycle.
- One precomputed fold/epoch manifest is consumed by both EEGNet and SIRE-EEG. Batch size remains 128 with full batches only.
- Architecture, optimizer, normalization, validation, early stopping, folds, outer-development evaluation, and metrics exactly match the frozen seed-0 reference.
- Scope: seed 0 only; no final-heldout data access, seed 1/2, or tuning.
