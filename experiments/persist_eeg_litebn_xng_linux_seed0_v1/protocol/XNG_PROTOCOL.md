# LiteBN-XNG protocol

- Model: LiteBN-XNG
- Seed: 0
- First task: WBCIC-MI
- Folds: canonical five-fold split already frozen by PERSIST-EEG
- Epochs: 60
- Optimizer: AdamW, learning rate 3e-4, weight decay 5e-4
- Gradient clipping: 5
- Loss: ordinary cross-entropy
- AMP: enabled on CUDA with the historical GradScaler behavior
- Selection: highest subject-equal inner-validation BA from epochs 10 through 60; strict improvement preserves the earliest tie
- Training batches: exact stored historical WBCIC-MI subject-disjoint episode manifests
- Normalization: exact stored fold normalizers fitted on inner-train source sessions

The XNG-only first pass trains and freezes all five candidate checkpoints before
outer-development evaluation. It does not train LiteBN and does not access the
internal heldout diagnostic. The continuation gate is therefore not evaluated
until the matched Linux LiteBN run exists.
