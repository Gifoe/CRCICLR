# WBCIC true-outer backbone re-evaluation protocol

This is an inference-only re-evaluation on the fixed, disjoint WBCIC final/true-outer cohort:

`sub-4, sub-8, sub-10, sub-15, sub-20, sub-39, sub-40, sub-43, sub-46, sub-51`

Only future session 2 is evaluated. Each subject is inferred by every one of the five frozen development-fold checkpoints. The development subjects, fold definitions, inner-selected epochs, training recipes, and fold-specific normalizers are unchanged. No true-outer example is used for training, checkpoint selection, tuning, calibration, model exclusion, or ensembling.

Evidence scope differs because only completed checkpoint families are eligible:

- EEGNet, TCFormer, CBraMod, TeCh: 5 folds x 3 seeds.
- LiteBN, ModernTCN, Medformer, SGN: 5 folds x seed 0 only.

The exact checkpoint hashes are locked before any true-outer array or label is loaded. Compact subject-, checkpoint-, and model-level metrics are retained; logits, labels, caches, and checkpoint binaries are not committed.
