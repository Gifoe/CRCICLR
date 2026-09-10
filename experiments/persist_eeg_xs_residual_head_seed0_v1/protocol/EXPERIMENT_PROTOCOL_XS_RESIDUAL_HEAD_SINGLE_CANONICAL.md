# LiteBN-XS zero-init residual head — seed0 canonical-inner protocol

Exact existing LiteBN-XS checkpoints are loaded for all four tasks and five folds. The complete XS network is frozen.
Only a zero-initialized residual classifier dW/db is trained on precomputed deterministic XS embeddings.
One frozen canonical inner_train_subjects / inner_val_subjects split is used per fold; no additional resplits.
max_epochs=30; patience=5; lr=0.0001; weight_decay=0.0005; beta=0.001; selection=inner-val subject-mean BA.
Epoch 0 is always legal and must exactly replay XS. No outer-development, final held-out/test, ensemble, gate/head fine-tuning, or embedding adapter is run.
frozen_litebn_x_source_code_sha256=5fa225c30f1938146a71f7655d56c487f457b8ab4a35c1a724a527e606c7dbf5
frozen_split_sha256=d313f8e1f2c105d0b387c607fb781b61d9a05d92875759e51ed0d4c6d889f42a
FINAL_HELDOUT_ACCESSED = NO
