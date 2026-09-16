# ModernTCN: four-task, five-fold, three-seed baseline

All 60 training cells and 60 frozen heldout checkpoint evaluations are complete. Values below are subject-equal balanced accuracy (mean ± sample SD across seeds), in percent. Each heldout seed averages the five frozen fold checkpoints; checkpoint selection and normalization used training/development data only.

| Task | Outer-development BA | Fixed heldout BA |
|---|---:|---:|
| OpenBMI MI | 61.65 ± 0.79 | 61.70 ± 0.32 |
| OpenBMI ERP | 77.80 ± 0.24 | 79.81 ± 0.12 |
| OpenBMI SSVEP | 85.53 ± 0.19 | 84.42 ± 0.48 |
| WBCIC MI | 71.14 ± 0.51 | 71.77 ± 0.39 |

The heldout evaluation used the post-training checkpoint lock and the established 14 OpenBMI and 10 WBCIC heldout subjects. The original monolithic evaluator encountered a Windows native access violation; an engineering-only recovery evaluated one frozen task/fold/seed checkpoint per fresh Python process, verified checkpoint and normalizer hashes, wrote resumable shards outside the repository, and aggregated with the original subject-equal formulas. No training, split, checkpoint, model, or metric was changed. See `code/baseline3_heldout_recovery.py`.

The complete row-level and seed-level results are in `outputs/`; `protocol/HELDOUT_LEAKAGE_AUDIT.json` records the heldout-access audit. Runtime shards, caches, raw EEG, and checkpoints are intentionally excluded from Git.
