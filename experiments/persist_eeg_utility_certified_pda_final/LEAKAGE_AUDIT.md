# Leakage audit

All source rows assert `future_session_used_for_fit=false`, `future_labels_used_for_fit=false`, and `future_labels_used_for_selection=false`. Shared basis fitting reads model-fit historical labels only. Historical held blocks are excluded from their own leave-one-block-out fit. S2, EEGNeX future, outer, sealed, utility_metrics, and utility_units were not read. Population archive identities remain unchanged.
