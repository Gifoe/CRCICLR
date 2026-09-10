# X-init residual adapter seed-0 protocol

Stage A only uses each frozen outer fold's historical inner-train plus inner-validation subjects. Outer-development and final held-out subjects are absent.

## Frozen global hierarchy

1. G0: adapter only; 2. G1: adapter plus X scale parameters with L2 anchor; 3. G2: adapter plus scale plus final mixer with L2 anchor.

Adapter: shared 2-8-1 MLP, x' = x * [1 + 0.5 * tanh(lambda_channel) * tanh(g)], lambda_channel=0 at initialization.
Optimization: AdamW lr=0.0001, weight_decay=0.0005, beta=0.0001, max_epochs=20, gradient_clip=5.0.
Inner partition base seeds: [101, 211, 307]; effective seeds additionally encode task and outer fold.
Checkpoint eligibility: median three-split BA delta > 0 and at least 2/3 deltas >= 0. Robust score = median - 0.5*std. If no eligible positive score, epoch 0 is selected.
Global hierarchy advance rule: a regime passes only if every task's mean selected robust gain is positive. No outer-development is opened otherwise.
Frozen source code SHA256: 2eef9a52fc533453a95718b2e3eee46fcfeebf3330058f3c22cda2d932359d9c
Frozen five-fold split SHA256: d313f8e1f2c105d0b387c607fb781b61d9a05d92875759e51ed0d4c6d889f42a
FINAL_HELDOUT_ACCESSED = NO
