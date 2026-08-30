# V1 reproduction audit

## Scope

`code/v1_reproduce.py` recomputes the V1 ATCNet-CleanRoom subject-balanced
statistics from the stored per-subject artifact. It does not load EEG data and
does not open any development, future, outer, or sealed resource.

## Pass condition

All quantities backed by the actual V1 code path must match the immutable
rounded values within `5e-6`: ERM, Mixup, RandomTransport,
SCST-NoConsistency, Full-SCST, Full-SCST minus ERM, its paired subject
bootstrap CI, fold sign count, and the consistency contribution.

## Explicit limitation

The prompt-specified `ShuffleSameClass` row is not present in the recovered V1
entrypoint, tracked results, runtime metric files, or logs. Its value is kept as
an immutable historical fact but is not called reproduced. This omission does
not alter the recovered primary Full-SCST test or any V2 implementation.

The machine-readable verdict is written to `results/V1_REPRODUCTION.json`.
