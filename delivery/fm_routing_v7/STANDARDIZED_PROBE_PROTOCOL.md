# Standardized probe protocol

Every frozen pooled embedding is followed by the same trainable `LayerNorm -> Linear` head. Outer fold `e` is evaluated once, `(e+1) mod 5` selects hyperparameters and epoch, and the other three folds train the head. Backbones never receive gradients.
