# LiteBN-XC protocol

- Candidate: LiteBN-XC only
- Base model: frozen Linux LiteBN-XNG
- Task first: WBCIC-MI
- Seed: 0
- Folds: canonical 0 through 4
- Epochs: 60
- Optimizer: AdamW, learning rate 3e-4, weight decay 5e-4
- Gradient clipping: 5
- Objective: ordinary cross-entropy
- AMP: FP16 with historical GradScaler behavior
- Selection: strict highest subject-equal inner-validation BA from epochs 10
  through 60; strict comparison preserves the earliest tie
- Batches: exact XNG WBCIC stored episode manifests
- Normalization: exact XNG fold normalizers

All five checkpoints are frozen before outer-development evaluation. Outer
analysis is frozen before the already-open internal heldout development
diagnostic is accessed. Gate diagnostics are observational and never influence
training or checkpoint selection.
