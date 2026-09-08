# Frozen method

- Tasks: OpenBMI ERP target/non-target and SSVEP four-frequency decoding.
- Participants: the frozen MI 40 SEARCH / 14 held-out OpenBMI membership.
- Sessions: source/cache session 1 to future/cache session 2.
- Carriers: canonical EEGNet and CompactLite-BN, initialized from scratch.
- Training: AdamW (`lr=3e-4`, `weight_decay=5e-4`), 60 epochs, batch 64,
  gradient clip 5.0, checkpoint selection by INNER_VAL future-session
  mean-subject BA from epochs 10--60.
- ERP loss: deterministic inner-train-S1 class-weighted CE, `N/(K*N_k)`.
- SSVEP loss: ordinary CE.
- Fusion: immutable equal-weight logit average, with no fitted fusion parameter.
- Analysis unit: subject after averaging its fixed replicates. Bootstrap uses
  10,000 subject resamples and seed 0.

The detailed, hash-locked protocol is in `protocol/`.