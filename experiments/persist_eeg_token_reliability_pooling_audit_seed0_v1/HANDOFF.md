# Current checkpoint — pre-result

The five task-trained CBraMod/OpenBMI_MI/seed-0 checkpoint preflights passed on
the original server. The token hook is `CBraModAdapter.model.encoder` output,
shape `(batch, 62, 4, 200)`. The benchmark's mean-pooling/native-head path
matched that hook on synthetic inputs. TRAIN Session-1 normalization hashes
and the five-fold split hash matched each checkpoint record. No OUTER or
formal-final-heldout EEG was opened in this audit.

Original-server repo:
`D:\nips-temp\TotalP\P1\CRCICLR_SEVEN_BACKBONE_HELDOUT_WORK`.
Runtime:
`D:\nips-temp\TotalP\P1\token_reliability_pooling_runtime`.
Access the server only via `D:\chenyu-iclr\remote_exec_safe.py`.

Three distinct scheduled tasks are active but currently waiting on earlier
stages/resources:

1. `PERSIST_EEG_TOKEN_RELIABILITY_TRAIN_EXTRACT_QUEUE_V1`: sequential TRAIN
   extraction for folds 0–4, using immutable server `run_v3.py`; requires
   >=40 GiB free RAM and >=16000 MiB free GPU before each new fold.
2. `PERSIST_EEG_TOKEN_RELIABILITY_ANALYZE_QUEUE_V1`: waits for all five
   extractions, then performs 500+500 reliability nulls and 200 subject
   split-halves per fold, writing runtime-only TRAIN source scores.
3. `PERSIST_EEG_TOKEN_RELIABILITY_UTILITY_QUEUE_V1`: waits for the previous
   task, then performs TRAIN-subject LOSO single-token and all-token-erasure
   utility analyses.

Do not duplicate these tasks or overwrite any partial output. Inspect their
exact task states, logs and runtime paths first. They are not a complete
experiment: TRAIN-only nested pooling selection, random matched controls,
OUTER evaluation after a TRAIN seal, subject-level statistics, cross-fold
stability, heatmaps, final report and reviewed GitHub push remain to be done.
The branch's current local commit is intentionally not a final result.

At this checkpoint, unrelated server work holds most GPU memory (~1 GiB
free), and RAM is ~30 GiB free. Do not stop unrelated jobs or lower the
heavy-launch gate merely to start this audit. Preserve every failure log;
earlier preflight implementation failures are documented in
`FAILURE_DISCLOSURES.md`.
