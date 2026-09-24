# Execution audit

- The first pre-heldout pilot used default linear-layer biases, contrary to the
  specified trainable W1 and W2 matrices. Its partial discovery/refit runtime
  was discarded. Its initial lock is preserved as
  PROTOCOL_LOCK_INVALID_BIAS.json for traceability. No formal heldout arrays
  were read in that run.
- The strict run used bias-free W1 and W2, a fresh runtime directory
  `/root/rivermind-data/native_gate_v2_strict_runtime`, and a fresh
  PROTOCOL_LOCK.json. All 20 discovery, 20 refit, and 20 formal heldout cells
  completed under that code. The final lock was written before the first
  formal heldout EEG array read.
- Pre-heldout synthetic checks passed: exact canonical EEGNet forward and
  zero-init logits, preserved successor complement, and five-fold aggregation
  with unequal fold projector ranks.
- After aggregation, the generated report's heuristic gain labels were
  corrected because they overstated zero/tiny BA differences. The original
  generated report is preserved as FINAL_REPORT_GENERATED_INITIAL.md. This
  reporting correction did not change the locked code, models, projectors,
  predictions, metrics, or bootstrap contrasts.
