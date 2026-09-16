# Medformer: four-task, five-fold, three-seed baseline

All 60 training cells and 60 frozen heldout checkpoint evaluations are complete. Values below are subject-equal balanced accuracy (mean ± sample SD across seeds), in percent. Each heldout seed averages the five frozen fold checkpoints; checkpoint selection and normalization used training/development data only.

| Task | Outer-development BA | Fixed heldout BA |
|---|---:|---:|
| OpenBMI MI | 70.17 ± 0.50 | 67.16 ± 0.39 |
| OpenBMI ERP | 78.91 ± 0.06 | 80.07 ± 0.42 |
| OpenBMI SSVEP | 91.87 ± 0.27 | 88.61 ± 0.10 |
| WBCIC MI | 75.13 ± 0.35 | 76.73 ± 0.13 |

The heldout evaluation used the post-training checkpoint lock and the established 14 OpenBMI and 10 WBCIC heldout subjects. After a Windows native access violation in the ModernTCN monolithic evaluator, the shared engineering-only recovery evaluated one frozen task/fold/seed checkpoint per fresh Python process, verified checkpoint and normalizer hashes, wrote resumable shards outside the repository, and aggregated with the original subject-equal formulas. No training, split, checkpoint, model, or metric was changed. See `code/baseline3_heldout_recovery.py`.

The complete row-level and seed-level results are in `outputs/`; `protocol/HELDOUT_LEAKAGE_AUDIT.json` records the heldout-access audit. Runtime shards, caches, raw EEG, and checkpoints are intentionally excluded from Git.
