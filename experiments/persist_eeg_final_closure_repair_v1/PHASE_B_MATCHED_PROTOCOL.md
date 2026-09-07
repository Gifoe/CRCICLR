# Phase B exact matched protocol

The primary comparison is PUD-Aux versus Matched-TaskOnly. Every method uses the same single-path EEGNet task network and training-only linear auxiliary head; Matched-TaskOnly sets lambda to zero. Within fold/seed/stage, complete initialization and task-network SHA, source rows, normalizer, minibatch order, optimizer, task loss, validation metric, patience, and epoch budget are identical.

Lambda grid: [0.05, 0.1, 0.25]. Inner teachers and certificates use inner_train only. Final teachers/certificates are rebuilt on all outer-source subjects only after `SELECTION_FROZEN.json` is written. Random targets are centered over classes before RMS matching to the legal PUD target. Outcome Session 2 is evaluated only after all selection artifacts are frozen.

OpenBMI internal holdout and WBCIC outer access are forbidden and guarded.
