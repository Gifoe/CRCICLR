# V1.2 design audit

V1.1 G2 failed because top-1 subject-ID BA is discrete and the 21-point
nearest-grid calibration selected alpha=0 for most Protected MEDIUM doses. V1.2
addresses only this measurement pathology. It does not change dataset, folds,
seeds, EEGNet representation, canonical spectrum, Protected blocks, V1.1
matched controls, task estimator, task metric, bootstrap unit, or outer lock.

The continuous metric is selected by the predeclared hierarchy, not by task
outcomes. The primary default is symmetric subject-ID log-loss/identity skill;
fallbacks are permitted only for train-only numerical pathology and must be
recorded in `IDENTITY_METRIC_AUDIT.md`. Controls are loaded and fingerprinted
from V1.1; no re-selection is legal.

The final terminal state will distinguish train infeasibility, held-out identity
transfer failure, unsupported Theory 3, and supported development evidence.
Because validation subjects were already used by V1/V1.1, this is a development
resource closure, not an untouched replication. No V1.3 is permitted for an
unfavorable scientific result.
