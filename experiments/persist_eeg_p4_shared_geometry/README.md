# PERSIST-EEG Shared Task Geometry Audit

This namespace contains the development-only Shared Task Geometry audit for the
PERSIST-EEG project.

- Development runs: folds `0,1,2` × seeds `0,1`.
- Data used: TRAIN and development validation only.
- Outer-test access: `false`.
- Protected assignments: frozen from Signed Audit V3; they were not redefined
  using geometry results.
- No PERSIST-USE model was trained.

## Current result

The recorded conservative status is `PERSIST_USE_MECHANISM_NOT_SUPPORTED`.
MI LOSO prototype transfer is positive, but MI is binary, so a class-RDM
Spearman correlation is mathematically undefined. The current report therefore
does not establish the required MI geometry gates. SSVEP RDM and
cross-session differences are below the matched random control.

`geometry_exception.txt` records the failed random-control continuation. The
JSON/CSV reports are retained as provenance and must not be interpreted as a
fully compliant pass of the audit. In particular, Gate D and the geometry-
utility link require a corrected rerun before any method design.
