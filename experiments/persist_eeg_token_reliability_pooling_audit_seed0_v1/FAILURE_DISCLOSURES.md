# Preserved pre-result implementation attempts

- `run_preflight_v1.ps1`: its direct Python call failed before producing a
  scientific result. The log is empty because the wrapper stopped on native
  stderr under PowerShell's terminating-error setting.
- `run_preflight_v2.ps1`: exposed the cause: `ModuleNotFoundError:
  backbone_models` in the new extraction entry point. Its transcript remains
  in the original-server runtime. The version-forward `run_v2.py` inserted the
  benchmark code directory before importing the adapter.
- `run_preflight_v3.ps1`: passed the fold-0 checkpoint, normalizer, split, and
  token-shape checks. Its status label incorrectly said `NO_EEG_READ` even
  though it recomputed the normalizer from TRAIN Session-1 EEG. No OUTER or
  final-heldout EEG was read. The label was corrected in `run_v3.py`.
- `run_preflight_all_v4.ps1`: passed all five folds using `run_v3.py`, and
  additionally checked equivalence of the token-hook mean/head path with the
  benchmark downstream forward on a synthetic input. It read TRAIN Session-1
  EEG only to recompute the benchmark normalizer.

These failures and corrections must not be presented as evidence for token
reliability or predictive utility. The heavy extraction and scoring queues
remain separate, hash-pinned, and fail closed on pre-existing outputs.

- The original TRAIN utility queue emits `ConvergenceWarning: Liblinear failed
  to converge` during high-dimensional all-token erasure fits. The queue is
  preserved rather than silently restarted or overwritten. The primary
  single-token utility probe does not use this high-dimensional fit; any
  erasure-based descriptive conclusion must be marked provisional unless a
  separately versioned convergence audit resolves the warning.

- The `PERSIST_EEG_TOKEN_RELIABILITY_OUTER_QUEUE_V1` scheduled task was stopped
  while it was only waiting for TRAIN selection. All five OUTER output paths
  were absent at the stop, and the partial V1 waiting transcript was kept.
  The V1 evaluator had draw-level random-control summaries but omitted the
  subject-level random-control rows needed for a biological-subject paired
  bootstrap. A version-forward V2 evaluator/queue adds those rows; V1 produced
  no scientific OUTER result and must not be restarted.
