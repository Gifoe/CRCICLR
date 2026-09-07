# PERSIST-EEG P4-SI: Selective Persistence Invariance

This directory contains the code and lightweight result artifacts from the
OpenBMI P4-SI development closure run on the designated GPU server.

## Scope

- SI-V0 through SI-V4 were evaluated on 3 development folds × 2 seeds each.
- All method selection used TRAIN/VALIDATION only.
- No raw EEG, cache, intermediate feature files, model checkpoints, tokens, or
  server credentials are included.
- The per-run files contain diagnostics, curves, and provenance only.

## Final decision

```text
P4_SELECTIVE_INVARIANCE_NOT_SUPPORTED
```

The experiments detect persistent and task-relevant structure, but no version
establishes the preregistered nuisance-suppression and validation
generalization requirements. Formal outer-test evaluation was not authorized;
the lock was refused.

See `results/P4_SI_FINAL_REPORT.md` and
`results/P4_SI_LOCK_REFUSED.json` for the closure decision. Version-specific
summaries are in `results/summaries/`.
