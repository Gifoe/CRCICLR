# PERSIST-EEG Phase 2.5: Prospective Utility Gate

This experiment asks whether source-only pseudo-target suppression utility ranks and predicts suppression utility on unseen outer subjects. It is a mechanism audit, not a new model.

## Frozen execution

```powershell
python code/freeze_protocol.py
python code/preflight.py
python code/scheduler.py --phase source
python code/freeze_source.py
python code/scheduler.py --phase outcome
python code/aggregate.py
python code/validate.py
```

The outcome phase refuses to run until all 30 source runs are frozen in `runtime/GLOBAL_SOURCE_FREEZE.json`. Runtime checkpoints/caches are ignored by Git.

## Result

Terminal state: `NO_ACTIONABLE_SUPPRESSION_HEADROOM`
Recommendation: `CLOSE_CONSTRUCTIVE_ROUTE`

See `FINAL_REPORT.md`, the audit Markdown files, `results/`, and `figures/`.
