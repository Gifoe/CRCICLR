# Trajectory measurement audit

Six frozen checkpoints were recorded: `t0, t1, t2, t3, t4, final`. At each checkpoint the script records P/U/D, identity control, history BA/loss, coordinate drift, protected contribution, and gradient-conflict quantities. Summary feature definitions are locked in `protocol/DYNAMIC_FEATURE_LOCK.json`.

Episodes measured: 40. All predictors use history-side quantities only; future S2 labels are used only to form the diagnostic target after the trajectory is complete.
