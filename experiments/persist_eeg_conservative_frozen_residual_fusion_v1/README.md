# CFRF-v1: Conservative Frozen Residual Fusion

This is the final planned constructive model-design experiment in the PERSIST-EEG line. It tests a bounded, sample-wise fusion correction around the validated EEGNet/LiteBN 50/50 logit fusion.

Both carriers are loaded from the frozen five-fold carrier checkpoints and are strictly evaluation-only. Head optimization uses only `INNER_VAL` source-session trials; `OUTER_DEV` labels are loaded only after all Phase-A cells are present. Runtime feature caches and checkpoints live outside the Git worktree.

Run `train_fusion_head.py --validate-only` before training, then complete all Phase-A cells, evaluate, and aggregate. A negative gate selects `FROZEN_LOGIT50` as the valid final fallback and ends model design.
