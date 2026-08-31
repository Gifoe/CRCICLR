# WBCIC S0→S1 action-bank audit

{
  "schema": "CGRFUSE_WBCIC_S0_S1_ACTION_BANK_V1",
  "source_sessions": [
    0,
    1
  ],
  "forbidden_sessions_materialized": [],
  "S2_accessed": false,
  "outer_accessed": false,
  "run_count": 6,
  "run_definition": "2 deterministic subject-grouped folds x 3 seeds",
  "actions": {
    "KEEP": "ERM source-trained EEGNet",
    "AMPLIFY": "source-trained subject-adversarial EEGNet (fixed GRL coefficient 0.10)",
    "GEOMETRY": "source-trained CORAL geometry-aligned EEGNet (fixed coefficient 0.10)"
  },
  "subjects": 41,
  "rows_per_session": {
    "0": 8198,
    "1": 8198
  },
  "training_labels_sessions": [
    0
  ],
  "evaluation_session": 1
}

The builder reads S0/S1 labels only and raises on any session 2 row. The action bank is six matched runs (2 subject folds × 3 seeds), with finite logits and exact subject/session/trial alignment.
