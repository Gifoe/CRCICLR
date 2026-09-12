# Frozen LocalStats seed0 protocol

Scope: OpenBMI_MI, OpenBMI_ERP, OpenBMI_SSVEP, WBCIC_MI; original five folds,
seed0 only. No ablations, tuning, additional seeds or sealed populations.

Use recovered, hash-verified historical CompactLite and its training functions.
An AST-checked insertion captures each branch immediately after spatial BN+ELU
and before its first AvgPool/dropout. Concatenate actual spatial BN dimensions
(48 in this implementation). Eight adaptive temporal bins supply local mean
and log1p(clamped second-moment variance), computed in float32 for numerical
stability. This is learned feature variation, not physiological band power.

Add bias-free q_dim->8->64 projection to the original embedding linear output;
retain original ELU, LayerNorm, dropout and single classifier. V uses ordinary
Linear initialization under independent subseed 741103; U is zero. Restore RNG
after module creation. No new convolution, normalization, dropout, loss or head.
The main forward is compiled from the actual historical function, not rewritten
from an architecture description. Every shared parameter and buffer is retained.

All 20 cells must pass shared-state, RNG, zero-U/nonzero-V and initial logits/
predictions equivalence gates in eval and training mode, FP32 and historical AMP,
before any training. Equal-RNG forward audit also checks BN buffer updates.
Restore all state after audit. Logit atol 1e-6, prediction mismatch exactly zero.

Reuse recovered exact seed0 initialization hashes, including Linux float32
uniform/serialization recovery already verified in the preceding experiment.
No pretrained baseline weights initialize candidate training. ERP and SSVEP
must preserve the historical constructor's extra head replacement RNG draws.

MI tasks retain historical support/query episode manifests, batch128, all60
epochs; normalizers and full deterministic manifests must match recorded hashes.
ERP/SSVEP retain session1 full-permutation batches64 and task/fold/epoch-specific
independent NumPy RNG. ERP uses original inverse-frequency weighted CE; SSVEP
ordinary multiclass CE. Training index hashes, class weights and normalizers
must match original provenance. Record all60 epoch batch-order hashes.

All tasks: original AdamW lr3e-4, wd5e-4, clip5, AMP/GradScaler, full60epochs,
canonical subject-equal validation BA, eligible epoch10..60, earliest tie
(strict improvement greater than1e-12). Dynamic AMP may skip overflowing steps;
record attempts and successful counts separately. No forced updates.

Train all20 cells before candidate outer evaluation, then the previously exposed
internal heldout diagnostic. Re-evaluate each original checkpoint with original
metric implementation and normalizer; require equality to historical per-subject
BA/F1/accuracy within1e-10. Fail closed if mismatch, rather than silently pooling
an unmatched baseline. Store source hashes and evaluation pairs. No new heldout.

Primary four-task decision uses equal-fold outer BA deltas. 4/4 strictly positive
is LOCALSTATS_SEED0_4OF4_SIGNAL. 3/4 with remaining task >=-0.5pp, positive equal
mean and no task <=-2pp is LOCALSTATS_SEED0_PARTIAL_SIGNAL. Otherwise use
LOCALSTATS_SEED0_NO_BROAD_SIGNAL. The -0.5pp near-zero and -2pp material-regression
thresholds are fixed before training/results. Tiny positive values are reported
literally, not called proven meaningful upgrades. Heldout is secondary diagnostic.
No automatic seed1/2 even for a positive terminal.

Report equal-task mean, median, worst, positive tasks/folds, paired subject
bootstrap10000 resamples seed0, all fold metrics and lower-level pairs. Readout
diagnostics (norms, ratio, q moments and separate mean/logvar contributions) use
selected checkpoints and evaluation forwards only; never select checkpoints.

Runtime limitation: available Windows CUDA execution is not original Linux
binary environment. Initial tensors and sampling are exactly verified, but this
is not proof of a bitwise identical full baseline trajectory across environments.
If baseline evaluation cannot be reproduced, record a blocker or obtain a matched
control; do not claim broad upgrade from an incompatible comparison.
