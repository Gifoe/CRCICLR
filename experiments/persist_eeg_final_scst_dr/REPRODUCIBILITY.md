# Reproducibility

- Parent commit: `57d5e4f1ae0a7c80d95ca27983fedad2ec3f690c`
- Server Python: `D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe`
- Development folds: the existing frozen OpenBMI 40-subject and WBCIC
  41-subject five-fold protocols.
- Model seed: historical ERM seed 0 for all folds/settings.
- Bootstrap: deterministic subject-cluster bootstrap, 10,000 resamples.
- Runtime features are not committed.  Compact metrics, hashes, reports, and
  figures are committed.

`protocol/PRE_STAGE0_FREEZE.json` records the exact code/protocol hashes before
transport outcomes.  `results/STAGE0_VALIDATION.json` independently checks the
freeze, cardinalities, gates, and sealed-resource state.

## Repair-1 hash lock

`protocol/PRE_STAGE0_REPAIR1_FREEZE.json` hashes the magnitude-only lock, all
three Repair-1 execution/analysis/validation programs, the unchanged common
implementation, and the validated V0 compact results before any Repair-1 unit
was computed.  Each of 40 Repair-1 units verifies its feature-scope, scaling
center, scaling scale, probe BA, and V0 unit hash before writing metrics.  The
raw pair-level Repair-1 CSVs remain under `runtime/stage0_repair1_units` and are
not committed; compact summaries, reports, and figures are committed.
