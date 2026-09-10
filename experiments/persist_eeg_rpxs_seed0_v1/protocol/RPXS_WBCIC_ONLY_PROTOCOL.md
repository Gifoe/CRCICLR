# RPXS WBCIC-only canonical-inner protocol

source_branch: `codex/persist-eeg-rpxs-seed0-v1`
source_commit: `d67133ae58e660368a6db5025f061c71abf98f8d`
model: LiteBN_RPXS = LiteBN_XS architecture initialized from LiteBN-X checkpoint.
X-isomorphic parameters copied from X; new channel_mlp and lambda_channel are new, with lambda_channel=0 for exact step0 replay.
residual-path dropout: batch-level Bernoulli p_gate_on=0.5 during training; inference gate ON.
optimizer: AdamW, new channel parameters lr=0.0001, X-initialized parameters lr=3e-05, weight_decay=0.0005.
L2-SP lambda=0.001; gate penalty beta=0.0001; max_steps=512; eval_steps=[0, 32, 64, 128, 256, 512]; clip=5.0.
selection: canonical inner-val BA; if no checkpoint strictly improves over step0 X, select step0.
canonical split sha256: `d313f8e1f2c105d0b387c607fb781b61d9a05d92875759e51ed0d4c6d889f42a`
scope: WBCIC_MI only; existing WBCIC fold0 record is reused and folds1-4 are resumed.
OpenBMI execution is intentionally not continued in this run.
OUTER_DEVELOPMENT_ACCESSED = NO
FINAL_HELDOUT_ACCESSED = NO
