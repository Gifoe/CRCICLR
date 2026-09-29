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

Status: all five source/checkpoint preflights and TRAIN token extractions
passed. The TRAIN reliability, 500+500 null, and 200 split-half audits are
complete. The hash-checked [TRAIN gate review](outputs/TRAIN_GATE_A_B_REVIEW.json)
finds that neither preregistered Gate A nor Gate B passes. This is an early
negative result, not evidence for pooling utility. The separate TRAIN utility
queue is still running; no OUTER or formal final-heldout EEG has been opened
for this audit. No predictive gate or final architecture decision is claimed.
See [the checkpoint findings](PRELIMINARY_TRAIN_FINDINGS.md) for exact values
and the limitations of this partial result.
