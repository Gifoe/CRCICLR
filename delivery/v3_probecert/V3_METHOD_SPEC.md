# ProbeCert-V3 method specification

## Target and access protocol

ProbeCert-V3 targets subject-level selection among No-TTA and frozen test-time interventions. Each episode is split chronologically into Adapt (A), Probe (P), and unchanged Future (V). Candidate actions update on A only and then call `freeze_state()`. Unlabeled P is used only for action diagnosis. V inputs and labels are inaccessible until the policy decision, action-state hash, configuration hash, and prediction-set index have been persisted and verified by the runtime access controller.

## Action library and Stage 0

The finite action library contains No-TTA, official T3A, and a corrected residual adapter. Every action starts from the same source checkpoint and is reinitialized per subject. The adapter uses a nonnegative collapse penalty. Stage 0 screens 64 T3A and 81 adapter configurations by subject-grouped successive halving on meta-development subjects, retains at most one configuration per action/dataset/seed, and measures label-informed Safe-Oracle headroom on development V. Oracle results are an upper bound, not deployment evidence.

## Probe policy

For each frozen action, the policy measures expected-set-size gain, nuisance-augmentation consistency, three-block temporal stability, source drift/class quality, and update magnitude on P. Thresholds and deterministic tie breaking are fit only on meta-fit subjects. The deployed policy selects an eligible action or falls back to No-TTA; no P labels, calibration outcomes, or outer outcomes enter this decision.

## Joint certificate

Each calibration subject contributes one policy-level critical index: the first lambda-grid index satisfying both future-risk control at alpha and noninferiority degradation at epsilon. The split-conformal order statistic uses `ceil((m+1)(1-delta))`; if this rank exceeds the calibration sample size, the method returns the full-set sentinel. Calibration occurs after the complete policy is frozen and contributes one scalar per subject, not one row per candidate action.

## Evaluation

Five seeded subject-level outer splits contain disjoint meta-fit, calibration, and outer-evaluation roles. Hyperparameter/action search and policy fitting use meta-fit subjects; calibration freezes the joint index; outer V is opened only after decisions are persisted. HMC and EEGMMIDB are development tasks. CAP transfers the HMC policy and action configurations and recalibrates only the conformal quantile at the target site; it is external-site replication, not untouched confirmation.
