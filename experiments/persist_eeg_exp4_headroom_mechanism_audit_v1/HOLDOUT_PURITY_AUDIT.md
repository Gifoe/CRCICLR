# Holdout purity audit

The historical 83.775% checkpoint is not legal. This audit reuses only the repaired 15-checkpoint family whose train/validation subject lists have zero overlap with the 14 internal-holdout subjects. `load_protocol()` filters metadata to V8_SEARCH before raw tensors are materialised. Target S2 is used only for diagnostic action scoring and prospective outcome evaluation; no target S2 quantity constructs an action or feature. Internal holdout accessed: **NO**.
