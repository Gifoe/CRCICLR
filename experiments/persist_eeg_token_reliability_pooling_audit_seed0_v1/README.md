# CBraMod pre-pooling token-reliability qualification audit

This new experiment tests whether cross-session task evidence varies across
channel-by-time tokens *before* CBraMod's global mean aggregation. It is not
another final-embedding Protected/complement intervention and does not train a
new neural backbone.

The exact seed-0 task-trained CBraMod/OpenBMI_MI selected checkpoints from the
seven-backbone SEARCH benchmark exist for all five frozen folds. Their backbone
states are frozen here. TRAIN_GEOMETRY alone defines reliability, utility,
ranking, selection, and classifier settings. CHECKPOINT_VALIDATION remains an
inventory-only, historically selection-exposed cohort. OUTER_DEVELOPMENT is
evaluation-only and historically exposed; formal final-heldout EEG must not be
opened. The design and stop rules are in `protocol/PROTOCOL_LOCK.json`.

This is a qualification audit, not an architecture build. A successful
heatmap alone cannot satisfy the decision gates. The previously observed
boundary is that pooled P/C structure can be diagnostically meaningful even
when final-embedding interventions do not robustly improve prediction.

Status: all five source/checkpoint preflights passed, including the exact
`62 × 4 × 200` hook shape and downstream-forward equivalence. Heavy TRAIN
token extraction, reliability nulls, and utility runs are queued behind a
server resource gate. No scientific result or OUTER evaluation is claimed
until all required outputs pass validation.
