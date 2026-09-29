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
The original high-dimensional erasure fit emits convergence warnings; preserve
the log and do not overinterpret U_erase. The extraction and reliability
queues are already complete and must never be rerun.

The preregistered qualification chain cannot pass Gate A or Gate B, regardless
of later predictive comparisons. The attachment nevertheless requested a full
selection/OUTER comparison; those stages have not been implemented or run.
Do not fabricate missing comparisons. A transparent early-stop report is
scientifically defensible; if continuing every comparison, complete TRAIN-only
selection before opening OUTER. Do not alter the frozen protocol or use OUTER
to rescue the negative premises.
