# GeoSR final constructive experiment

GeoSR (Geometry-Guided Subject-Robust Learning) is the final pre-registered
constructive attempt for PERSIST-EEG.  It uses the canonical EEGNet unchanged
and only changes source-subject loss weights.  Geometry and source difficulty
are computed by 5-way subject cross-fitting inside each canonical fold; no
raw latent vectors are compared across teachers.

Execution on the authorized server:

```text
<GPU-python> code/run_geosr.py --phase all --seed 0 --device cuda
<GPU-python> code/validate.py
```

The runner writes a pre-outcome lock after all source weights and final-refit
checkpoints exist.  Outcome labels are loaded only after that lock.  If the
frozen seed-0 gates fail, the terminal is `GEOSR_FINAL_CONSTRUCTIVE_STOP` and
no seed 1/2 or rescue search is allowed.  Runtime tensors and checkpoints are
ignored by git; compact tables, locks, code, and reports are deliverables.
