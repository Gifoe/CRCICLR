# LiteBN-DRA-XS seed0 protocol

Exact LiteBN-XS architecture from random seed 0; no trained XS/X checkpoint is loaded.
Single intervention: admission of the existing channel residual.
Admission: epochs 1-5 scale 0; epochs 6-9 scale (epoch-5)/5; epoch 10+ scale 1.
At scale 0 the channel MLP and lambda_channel path are bypassed completely; all other XS parameters train normally.
AdamW lr=0.0003; weight_decay=0.0005; clip=5.0; max_epochs=60; min_selection_epoch=10; patience=10.
Selection: canonical inner-validation subject-mean BA; ties use higher macro-F1 then earlier epoch.
Stage 1 opens OpenBMI ERP only. WBCIC MI opens only after the frozen ERP clear-signal rule.
ERP clear signal: mean DRA-vs-XS >= +0.20 pp and at least 3/5 positive folds.
canonical split sha256: `d313f8e1f2c105d0b387c607fb781b61d9a05d92875759e51ed0d4c6d889f42a`
SEED = 0
CANONICAL_INNER_SPLITS_ONLY = YES
NEW_INNER_SPLITS_CREATED = NO
OUTER_DEVELOPMENT_ACCESSED = NO
FINAL_HELDOUT_ACCESSED = NO
