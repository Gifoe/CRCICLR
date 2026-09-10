# X-init residual adapter seed-0 — canonical-inner G2 WBCIC-only protocol

Only WBCIC_MI is evaluated: five frozen outer folds, one canonical inner train/validation pair per fold.
No additional resplits are created. OpenBMI, G1, G0, G2 follow-up, outer-development and final held-out evaluation are not run.

Epoch 0 is the exact LiteBN-X checkpoint; tau=0.5; adapter lambda_channel=0 at initialization.
Trainable modules: residual_adapter.lambda_channel, residual_adapter.channel_mlp, X.scale_mlp, X.lambda_scale, X.mixer[-1]. All remaining LiteBN-X parameters and BatchNorm buffers are frozen.
AdamW lr=0.0001, weight_decay=0.0005, beta=0.0001, max_epochs=20, patience=5, selection metric=inner-val BA.
Best checkpoint is saved after each strict BA improvement. Training stops after five consecutive non-improving epochs. If no epoch improves over epoch 0, epoch 0 is selected.
Frozen source code SHA256: 7ef2cf69c4dd03b2c1b94b30de187a3b7ccb814c47eb50b9492d117c5f6fa94e
Frozen five-fold split SHA256: d313f8e1f2c105d0b387c607fb781b61d9a05d92875759e51ed0d4c6d889f42a
FINAL_HELDOUT_ACCESSED = NO
