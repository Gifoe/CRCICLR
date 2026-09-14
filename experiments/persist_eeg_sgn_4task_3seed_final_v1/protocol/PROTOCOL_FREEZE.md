# SGN matched baseline protocol freeze

- Model: official `colison/SGN` release at commit `c6d1b573dcb8c4255cde59b988f74334ea5da503` linked in the NeurIPS 2025 paper.
- Source gate: VGE, MGWM and PWSM are all present. The fixed official PTB-XL standard configuration is used across tasks; only channel count, sequence length and class count vary.
- Tasks: OpenBMI MI, ERP, SSVEP and WBCIC MI.
- Replicates: frozen folds 0--4 and seeds 0, 1, 2; exactly 60 independent models.
- Subject/session roles: OpenBMI inner-train session 1, inner-validation/outer-development session 2. WBCIC inner-train internal sessions 0 and 1, inner-validation/outer-development internal session 2.
- Normalization: channelwise mean and standard deviation fitted only on source-session inner-training samples, then applied unchanged.
- Optimization: 60 epochs, physical batch 128, AdamW, learning rate 3e-4, weight decay 5e-4, global gradient clipping 5.0, no scheduler, no early stopping, no augmentation.
- Loss: CRCICLR baseline CE only, weighted by inverse training-class frequency for OpenBMI ERP. The optional upstream similarity auxiliary term is not added.
- Selection: epochs 10--60 only; maximum future-session inner-validation subject-equal balanced accuracy; earliest epoch on exact ties.
- Outer development: selected checkpoint evaluated once on corresponding disjoint outer-development subjects and never used for selection.
- Heldout: inaccessible during fitting and selection. Evaluation is permitted only after all 15 checkpoint hashes for a task are frozen and must reuse the established CRCICLR evaluator and aggregation rule.

