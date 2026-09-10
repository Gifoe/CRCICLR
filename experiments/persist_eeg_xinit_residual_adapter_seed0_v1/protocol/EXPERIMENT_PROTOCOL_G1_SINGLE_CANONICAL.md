# X-init residual adapter seed-0 — canonical-inner G1 protocol

Stage A uses exactly each frozen fold's historical inner_train_subjects and inner_val_subjects. No additional resplits are created.
Only XRA_G1_ADAPTER_SCALE is run in this amendment. G0/G2 are not run and no outer-development is opened.

Epoch 0 is the exact LiteBN-X checkpoint; tau=0.5; adapter lambda_channel=0 at initialization.
Trainable modules: residual_adapter.lambda_channel, residual_adapter.channel_mlp, X.scale_mlp, X.lambda_scale. All remaining LiteBN-X parameters and BatchNorm buffers are frozen.
AdamW lr=0.0001, weight_decay=0.0005, beta=0.0001, max_epochs=20, patience=5, selection metric=inner-val BA.
Best checkpoint is saved after each strict BA improvement. Training stops after five consecutive non-improving epochs. If no epoch improves over epoch 0, epoch 0 is selected.
Target summary: OpenBMI MI/ERP/SSVEP non-negative versus X; WBCIC MI positive mean delta. This run stops after the 20 G1 inner runs for inspection.
Frozen source code SHA256: 89c712dbd6f2e59f26117648c56f6f7a36c0420e964cf09d7c99b5cbcd920da1
Frozen five-fold split SHA256: d313f8e1f2c105d0b387c607fb781b61d9a05d92875759e51ed0d4c6d889f42a
FINAL_HELDOUT_ACCESSED = NO
