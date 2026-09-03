# Frozen protocol

## Scope

Run only seed `0`, datasets OpenBMI and WBCIC, folds `0,1,2,3,4`. Reuse the
canonical EEGNet seed-0 checkpoint, normalizer, PSG V2 AdamW settings,
gradient clipping, dropout/RNG schedule, BN handling, and A/meta-fold
construction. No PSG correction is applied and no candidate is trained.

## Batches

For each source/refit biological subject `s`, deterministically permute legal
source trials independently per class and split five blocks: `B_s_1` through
`B_s_4` are certificate blocks and `B_s_out` is held-out. Blocks are mutually
trial-disjoint and class-balanced. Before any harm outcome is measured,

`m_per_class = min(16, floor(minimum_available_per_class / 5))`,

with a hard minimum of 4 and no replacement. The implementation uses the
resulting dataset-specific value and records it in the preflight artifact.

## Certificate and harm

At `theta_t`, compute eval-mode, dropout-free subject gradients `g_s_1...g_s_4`.
For `K in {1,2,4}`, `gbar_s_K` is their arithmetic mean and
`c_same = gbar_s_K^T Delta_A`. `Delta_A` is the exact measured displacement
from one PSG V2 task-only A gradient followed by AdamW. Held-out harm is
`H_s = L(B_s_out; theta_t + Delta_A) - L(B_s_out; theta_t)` and its sign is the
harm label. K=4 is the predeclared primary; K=1 and K=2 are diagnostics.

## Controls and inference

The different-subject control uses a deterministic cyclic partner from the
same B meta-fold, hence it is A-disjoint. The permutation control is a
deterministic non-self derangement. The random control is a deterministic
Gaussian direction with exactly the same norm as `gbar_s_K`. Pooled B is only
a group diagnostic. Biological subject is the inference unit. Observation
metrics are descriptive; 10,000-draw cluster bootstrap resamples subjects,
not individual steps. Undefined AUROC remains undefined.

## Compute schedule and gate

To bound runtime, five evenly spaced trajectory steps per dataset/fold are
selected by `linspace(1, total_steps, 5)` before outcomes and are recorded in
`runtime/PREFLIGHT.json`. This is a compute-budget subsampling rule, not an
outcome selection. Strong pass requires the exact two-dataset K=4 gate from
the task specification: AUROC >= .60 with cluster CI lower > .50, positive
Spearman with lower CI > 0, positive same-minus-different AUROC and Spearman
advantages with lower CIs > 0, and at least 4/5 non-negative fold AUROC
advantages, plus all legality/implementation checks.

No result can authorize Step 2 automatically.
