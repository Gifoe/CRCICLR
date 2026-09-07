# Leakage-separated table schema

The formal pipeline uses eight tables with an explicit decision/outcome boundary.

1. `subject_context_features.parquet`: `U_s`-only embeddings, probability summaries, channel mask, signal quality, and instability.
2. `action_context_diagnostics.parquet`: `U_s`-only per-action adaptation and context prediction-set diagnostics.
3. `historical_action_outcomes.parquet`: future outcomes for meta-risk training, conformal calibration, or CAP target-site calibration only.
4. `critical_index_predictions.parquet`: alpha-specific predicted critical indices and model hashes.
5. `certified_action_candidates.parquet`: conformal corrections, certified indices, selected lambdas, `n_classes`, and context-only utility.
6. `pre_outcome_decisions.parquet`: final-test decisions frozen before any future outcome is evaluated.
7. `final_test_outcomes.parquet`: offline future outcomes created only after the freeze gate passes.
8. `subject_decisions.parquet`: one-to-one join of the preceding decision and outcome tables.

Formal Pydantic rows forbid undeclared fields. Context and pre-outcome schemas reject future metrics. All curves, aggregation, joins, and validation use the complete identity key: dataset, seed, episode, subject, action, and alpha as applicable.
