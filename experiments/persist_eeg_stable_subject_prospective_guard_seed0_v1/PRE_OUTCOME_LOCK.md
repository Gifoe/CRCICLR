# PERSIST-SSPG pre-outcome lock

This lock was written after source-only checkpoint, schedule, legality and mandatory engineering tests and before any development outcome index or label was materialized. The machine-readable lock is `results/PRE_OUTCOME_LOCK.json`.

- code_commit: 23c9593db347760018601febf5327c2135f76e07
- K=4; m_per_class=16; continuation=2 epochs; kappa=0.20
- optimizer: AdamW, lr=3e-5, weight_decay=5e-4, clip=5
- parameter scope: full trainable parameter space; BN running statistics frozen
- outcome access before lock: false
- WBCIC outer-10 opened: false
- OpenBMI sealed/confirmation opened: false
- seed1_run: false; seed2_run: false

No development outcome evaluation may run unless this file and JSON are present and committed.
