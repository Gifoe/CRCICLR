# Execution repairs (not model or outcome tuning)

1. Initial Windows checkpoint search identified incompatible smaller
   seven-backbone ERP/SSVEP models. The user supplied the historical Linux
   server; 60 selected checkpoints and 101 source/protocol/result files were
   transferred and independently SHA256 verified on Windows. No raw EEG copy.
2. Windows SSH child-process lifetime prevented an initial detached launch from
   persisting. The active job instead runs in a retained SSH command session;
   stdout/stderr are written to dedicated log files. Existing PID 25168 was
   never terminated.
3. First replay used an unnecessarily disabled cuDNN TF32 flag. All 20 outer
   cells matched exactly, but ERP fold0 internal-heldout subject12 differed by
   one prediction. Historical evaluation code does not disable cuDNN TF32.
   Restoring `cudnn.allow_tf32=True` reproduced all three historical metrics
   for this subject exactly. The difference was not a checkpoint/label change:
   original subject BA was 0.9603030303030302; TF32-disabled BA was
   0.9606060606060606. The initial runtime is preserved and the corrected run
   replays all cells into a new invariant-bound runtime before calibration.
   No BN calibration occurred before this repair. No outcome threshold was
   relaxed. The same corrected inference flags apply to original and Precise-BN.
4. A post-transfer attempt to reconnect to the Linux host failed at the TCP
   connection stage. All required artifacts had already been transferred and
   hash-verified; Windows execution does not depend on continued source access.
