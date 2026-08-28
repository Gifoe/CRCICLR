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

