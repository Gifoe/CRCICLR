# SGN: four-task, five-fold, three-seed baseline

All 60 training cells and 60 frozen heldout checkpoint evaluations are complete. Values below are subject-equal balanced accuracy (mean ± sample SD across seeds), in percent. Each heldout seed averages the five frozen fold checkpoints; checkpoint selection and normalization used training/development data only.

| Task | Outer-development BA | Fixed heldout BA |
|---|---:|---:|
| OpenBMI MI | 51.92 ± 0.43 | 52.26 ± 0.11 |
| OpenBMI ERP | 73.72 ± 0.30 | 75.22 ± 0.21 |
| OpenBMI SSVEP | 29.06 ± 1.53 | 29.97 ± 2.41 |
| WBCIC MI | 67.10 ± 0.82 | 68.08 ± 0.71 |

The heldout evaluation used the post-training checkpoint lock and the established 14 OpenBMI and 10 WBCIC heldout subjects. After a Windows native access violation in the ModernTCN monolithic evaluator, the shared engineering-only recovery evaluated one frozen task/fold/seed checkpoint per fresh Python process, verified checkpoint and normalizer hashes, wrote resumable shards outside the repository, and aggregated with the original subject-equal formulas. No training, split, checkpoint, model, or metric was changed. See `code/baseline3_heldout_recovery.py`.

The complete row-level and seed-level results are in `outputs/`; `protocol/HELDOUT_LEAKAGE_AUDIT.json` records the heldout-access audit. Runtime shards, caches, raw EEG, and checkpoints are intentionally excluded from Git.
