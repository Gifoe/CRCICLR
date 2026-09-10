# BPXS first-screen protocol

model: LiteBN_BPXS = exact original random-initialized LiteBN_XS architecture.
Training objective: L_on + alpha*L_off + gamma*KL(stopgrad(p_off)||p_on).
seed: 0; alpha: 0.5; gamma: 0.1; temperature: 1.0.
optimizer: AdamW lr=0.0003; weight_decay=0.0005; clip=5.0; max_epochs=60; selection epochs >= 10; patience=5.
Gate-off disables only the XS channel residual correction; the scale gate, stem, mixers, embedding, classifier and dropout remain shared.
Scope: OpenBMI ERP and WBCIC MI, five canonical inner splits each; no MI/SSVEP expansion.
canonical split sha256: `d313f8e1f2c105d0b387c607fb781b61d9a05d92875759e51ed0d4c6d889f42a`
CANONICAL_INNER_SPLITS_ONLY = YES
OUTER_DEVELOPMENT_ACCESSED = NO
FINAL_HELDOUT_ACCESSED = NO
