# Frozen benchmark protocol audit

The baseline-metrics closure is the final metric and true-outer cohort authority. Its `SESSION_MAPPING_AUDIT.json` maps OpenBMI file S1/S2 to paper S1/S2 and WBCIC file S0/S1/S2 to paper S1/S2/S3; its `BOOTSTRAP_PROTOCOL.json` fixes 20,000 biological-subject draws and aggregation before bootstrap. The seven-backbone four-task `benchmark_data.load_search_fold` is the matched **training** source: frozen subject folds, source-session training, future-session inner validation, outer-development evaluation, task label map, and normalization fitted on inner source training only. These roles are complementary, not interchangeable.

WBCIC final true-outer subjects are `sub-4, sub-8, sub-10, sub-15, sub-20, sub-39, sub-40, sub-43, sub-46, sub-51`. V8 internal IDs are not used as final true outer. OpenBMI final heldout subjects are `4, 12, 13, 17, 18, 24, 25, 29, 36, 37, 39, 42, 51, 54`. Neither cohort is opened before the complete 120-checkpoint pre-evaluation lock.

Optimization uses the most recent common baseline recipe: AdamW, learning rate 3e-4, weight decay 5e-4, batch 128, gradient clipping 5, maximum 60 epochs, first selectable epoch 10, eight eligible no-improvement epochs to stop. Inner-validation subject-equal BA chooses the checkpoint. Outer development is reporting only, not selection. ERP alone uses the benchmark's class-frequency-weighted CE.

The formal SIRE-EEG is a LiteBN from another server, not this server's closure LiteBN. A pairwise comparison will be reported only if its exact subject/session true-heldout records can be verified and aligned.
