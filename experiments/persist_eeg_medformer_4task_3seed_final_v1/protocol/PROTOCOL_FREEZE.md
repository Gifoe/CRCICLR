# Medformer matched baseline protocol freeze

- Model: official paper-version `DL4mHealth/Medformer` at commit `446275f27b713a9f09917a6ba0bc51a18e921597`.
- Architecture: fixed subject-independent TDBRAIN setting requested in the experiment specification; only channel count, sequence length, and class count vary by task. Augmentation and SWA are disabled.
- Tasks: OpenBMI MI, ERP, SSVEP and WBCIC MI.
- Replicates: frozen folds 0--4 and seeds 0, 1, 2; exactly 60 independent models.
- Subject/session roles: OpenBMI inner-train session 1, inner-validation/outer-development session 2. WBCIC inner-train internal sessions 0 and 1, inner-validation/outer-development internal session 2.
- Normalization: channelwise mean and standard deviation fitted only on source-session inner-training samples, then applied unchanged.
- Optimization: 60 epochs, physical batch 128, AdamW, learning rate 3e-4, weight decay 5e-4, global gradient clipping 5.0, no scheduler, no early stopping, no augmentation, no auxiliary objective.
- Loss: ordinary cross entropy except inverse-frequency weighted cross entropy for OpenBMI ERP, with weights derived only from legal training labels.
- Selection: epochs 10--60 only; maximum future-session inner-validation subject-equal balanced accuracy; earliest epoch on exact ties.
- Outer development: selected checkpoint evaluated exactly once on its disjoint outer-development subjects and never used for selection.
- Heldout: inaccessible during fitting and selection. Evaluation is permitted only after all 15 checkpoint hashes for a task are frozen and must reuse the established CRCICLR evaluator and aggregation rule.

