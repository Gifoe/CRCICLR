# Frozen leakage-separated GPU/statistics interface

GPU computation must use the fixed `episodes_main120`, internal splits, and `CHANNEL_PROTOCOL.json`. It may write embeddings and logits for all roles as a frozen transformation, but it must not use final-test performance for fitting, hyperparameter selection, calibration, or decisions.

The interface consists of eight Parquet tables:

1. `subject_context_features.parquet` — U-only features.
2. `action_context_diagnostics.parquet` — U-only action diagnostics.
3. `historical_action_outcomes.parquet` — V outcomes for permitted historical/calibration roles only.
4. `critical_index_predictions.parquet` — alpha-specific predictions and model hashes.
5. `certified_action_candidates.parquet` — certified indices and U-only utility.
6. `pre_outcome_decisions.parquet` — immutable U-only final-test decisions.
7. `final_test_outcomes.parquet` — future outcomes generated only after the freeze gate passes.
8. `subject_decisions.parquet` — one-to-one offline join of decisions and outcomes.

The selector accepts table 5 only and rejects future fields. Table 6 must be written and hash-locked before table 7 can be computed. `verify_final_test_gate` must pass immediately before any final-test V labels are opened. Schema definitions are in `hsc_tta.schemas.models`; the theoretical definitions are in `docs/THEORY_SPEC.md`.
