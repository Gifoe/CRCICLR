# Current checkpoint — TRAIN reliability premises failed

Original server access is only via `D:\chenyu-iclr\remote_exec_safe.py`.
Original-server runtime is
`D:\nips-temp\TotalP\P1\token_reliability_pooling_runtime`.

All five frozen CBraMod/OpenBMI_MI/seed-0 TRAIN token extractions and all five
reliability/null/split-half audits are complete. The local reviewed artifact
`outputs/TRAIN_GATE_A_B_REVIEW.json` is byte-identical to the original-server
runtime artifact (SHA256
`a8688d553ab007ff5d4926c493171e63f226d5fe32c3c0fb11aed720429701aa`).
Its source analysis script is `code/audit_train_gates.py` (deployed as
`audit_train_gates_v1.py`). Every source seal and declared file hash was
checked before calculation. Gate A fails in all five folds: each real
reliability standard deviation is below the subject-pairing-null p95. Gate B
also fails numerically: across-fold mean split-half Spearman is 0.275 versus
the frozen 0.50 threshold and mean top-quartile Jaccard is 0.231 versus 0.40.
No OUTER or formal final-heldout EEG has been read for this audit.

The exact scheduled task `PERSIST_EEG_TOKEN_RELIABILITY_UTILITY_QUEUE_V1` is
still processing folds sequentially. Inspect its state and
`utility_queue_v1.log` before taking action; do not duplicate or overwrite it.
At this checkpoint folds 0 and 1 have utility seals; fold 2 is processing.
The distinct light task
`PERSIST_EEG_TOKEN_RELIABILITY_TRAIN_EVIDENCE_QUEUE_V1` is registered and
waiting for the utility task to finish with result 0. It uses hash-pinned
`code/summarize_train_evidence.py` deployed as
`summarize_train_evidence_v1.py`, creates runtime-only `train_evidence_v1`,
and must not be duplicated. Validate its output hashes and copy only compact
tables to the worktree after completion.

The exact task `PERSIST_EEG_TOKEN_RELIABILITY_SELECTION_QUEUE_V1` is also
running in a *waiting* state. It waits for TRAIN evidence success, then uses
`code/select_train.py` (remote `select_train_v1.py`) to recompute R and U
without each held TRAIN biological subject, select q/tau/C entirely inside
TRAIN, and create five immutable `TRAIN_SELECTION_SEAL.json` files. This is
computationally expensive; do not start another heavy lane or duplicate it.

The original `PERSIST_EEG_TOKEN_RELIABILITY_OUTER_QUEUE_V1` waiting task was
stopped before any OUTER output because its random-control summary omitted
subject-level rows needed for paired bootstrap. Its transcript is preserved.
The version-forward task `PERSIST_EEG_TOKEN_RELIABILITY_OUTER_QUEUE_V2` is
running in a *waiting* state; it may open OUTER only after all five TRAIN
selection seals and a successful selection task. It then extracts/evaluates
one fold at a time, with 500 random top-k and 500 random two-stream controls
and subject-level rows. V1 must not be restarted.

The light `PERSIST_EEG_TOKEN_RELIABILITY_FINAL_COMPACT_QUEUE_V1` is waiting
for OUTER V2 success. It hash-checks every source seal, runs 20,000 paired
biological-subject bootstrap draws, generates required tables/heatmaps,
decision JSON, and report into runtime-only `final_compact_v1`. Review every
file and hash before copying compact artifacts to GitHub. Any stage failure
must be preserved and corrected under a new version, never overwritten.
The original high-dimensional erasure fit emits convergence warnings; preserve
the log and do not overinterpret U_erase. The extraction and reliability
queues are already complete and must never be rerun.

The preregistered qualification chain cannot pass Gate A or Gate B, regardless
of later predictive comparisons. The attachment nevertheless requested a full
selection/OUTER comparison; code and fail-closed queues are staged but those
results have not run. Do not fabricate missing comparisons. Do not alter the
frozen protocol or use OUTER to rescue the negative premises.
