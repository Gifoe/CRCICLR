# Native C-to-P Transfer Gate V2 (EEGNet, seed 0)

This experiment uses the canonical four-task, five-fold EEGNet protocol. The
EEGNet weights and BN state are frozen during gate training. Spatial and
successor orthogonal Protected projectors are fitted only on legal development
trials using the audited PathFit ridge coefficient 1.0. An equal-rank seeded
random projector is fixed before final evaluation.

The gate scales the actual frozen nonlinear complement-to-Protected transfer
between the two stages. It is zero-initialized to exactly recover EEGNet.
BASELINE, P_ONLY_TRANSFER_GATE, RANDOM_TRANSFER_GATE, and
PROTECTED_NATIVE_TRANSFER_GATE all enter formal heldout evaluation.

Run stages in order: preflight; discover for all 20 task/fold cells; refit for
all 20 cells; lock; final-eval for all cells; aggregate. The final heldout
loader checks FINAL_EVAL_LOCK before opening EEG arrays. Outputs live in
outputs/ and locks in protocol/. Large checkpoints remain in the runtime
directory specified by NATIVE_GATE_RUNTIME.
