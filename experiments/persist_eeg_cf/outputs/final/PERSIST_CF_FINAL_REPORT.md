# PERSIST-CF final report

Terminal state: `PERSIST_CF_NOT_SUPPORTED`.

The structural mechanism is valid but the learning result is not useful.  The
six TRAIN-only audits retained 73.9%--90.1% of subject-offset energy in
Protected $G^\perp$, while keeping the relative geometry-projection error
below `1.093e-07`.

CF0 improved natural BA by only `+0.000496`
and stress robust BA by `+0.000271`.
CF1-HARD changed these to `-0.000032`
and `+0.000411`.  Both robustness
effects are far below the `+0.010` continuation threshold and the `+0.020`
ROBUSTNESS_PASS threshold.  Duplicate-clean explains most of CF0's tiny
natural change.

No method lock was created.  Development validation, outer test, ERP, SSVEP,
and EEGMMIDB were not evaluated.  Continuing with another refinement would be
unsupported search rather than a protocol-driven adjustment.
