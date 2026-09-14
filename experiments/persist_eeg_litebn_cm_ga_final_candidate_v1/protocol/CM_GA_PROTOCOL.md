# LiteBN-CM-GA locked protocol

The full protocol is the user-supplied `LiteBN-CM-GA` final-candidate prompt.
This implementation does not conduct architecture or hyperparameter search.

Stage 0 uses seed 0, all five canonical folds, inner-training biological
subjects only, and source-session trials only. For every subject pair A/B,
`A_grad`, `B_guard`, and `B_future` are class-balanced; A and B are distinct;
and `B_guard` and `B_future` contain disjoint trial IDs. Two deterministic
pairing rounds are used, with eight trials per class per block. The randomized
different-guard control substitutes a deterministic third biological subject.

The optimizer-equivalent candidate update is one initial-state AdamW update at
learning rate 3e-4 and weight decay 5e-4 after global gradient clipping at
5.0. It is applied to an in-memory residual state and then bitwise restored.
Base parameters and BN buffers remain frozen and in eval mode.

The biological subject is the cluster-bootstrap unit. Ten thousand draws are
used. Each dataset passes only when real AUROC is at least 0.60 with lower 95%
CI above 0.50, real Spearman and its lower CI are positive, and both AUROC and
Spearman exceed the different-guard control with positive lower confidence
bounds. Both datasets must pass before Stage 1 is authorized.

