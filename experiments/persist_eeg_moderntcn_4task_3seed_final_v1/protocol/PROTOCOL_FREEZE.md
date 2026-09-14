# ModernTCN matched baseline protocol freeze

- Model: official `luodhhh/ModernTCN` classification implementation at commit `56a9a2c018385cd5acef015378cae7f084d1b11c`.
- Architecture: the fixed official SelfRegulationSCP2 configuration requested in the experiment specification; only channel count, sequence length, and class count vary by task.
- Tasks: OpenBMI MI, ERP, SSVEP and WBCIC MI.
- Replicates: frozen folds 0--4 and seeds 0, 1, 2; exactly 60 independent models.
- Subject/session roles: OpenBMI inner-train session 1, inner-validation/outer-development session 2. WBCIC inner-train internal sessions 0 and 1, inner-validation/outer-development internal session 2.
- Normalization: channelwise mean and standard deviation fitted only on source-session inner-training samples, then applied unchanged to all other samples.
- Optimization: at most 60 epochs, physical batch 128, AdamW, learning rate 3e-4, weight decay 5e-4, global gradient clipping 5.0, no scheduler, no augmentation, no auxiliary objective. Train through epoch 10, then stop after 8 consecutive epochs without a strict improvement in inner-validation subject-equal balanced accuracy.
- Loss: ordinary cross entropy except inverse-frequency weighted cross entropy for OpenBMI ERP, with weights derived only from legal training labels.
- Selection: epochs 10--60 only; maximum future-session inner-validation subject-equal balanced accuracy; earliest epoch on exact ties.
- Outer development: selected checkpoint evaluated once on the corresponding disjoint outer-development subjects. Outer-development metrics never alter selection.
- Heldout: inaccessible during fitting and selection. It may be evaluated only after all 15 checkpoint hashes for a task are frozen, using the already established CRCICLR evaluator and aggregation rule.
