# Source-only Precise-BN seed0 final diagnostic

PRECISE_BN_NO_USEFUL_SIGNAL

Recovered checkpoints supersede the initial source-gate blocker. All 20 original checkpoints replayed against historical per-subject BA, macro-F1 and accuracy before any recalibration.

| Task | Original outer BA | Precise outer BA | Delta pp | Original heldout BA | Precise heldout BA | Delta pp | Fold SD change pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | 0.791500 | 0.788000 | -0.350 | 0.744857 | 0.744857 | +0.000 | -0.117 |
| OpenBMI_ERP | 0.832553 | 0.831568 | -0.098 | 0.846545 | 0.845623 | -0.092 | -0.026 |
| OpenBMI_SSVEP | 0.892000 | 0.891250 | -0.075 | 0.907000 | 0.907571 | +0.057 | -0.106 |
| WBCIC_MI | 0.787299 | 0.710899 | -7.640 | 0.792200 | 0.729000 | -6.320 | +7.773 |

Outer equal-task delta: -2.041 pp; positive tasks: 0/4; median -0.224; worst -7.640.
Internal heldout equal-task delta: -1.589 pp; positive tasks: 1/4; median -0.046; worst -6.320.
Improved folds: 6/20; mean fold-SD change: +1.881 pp (population SD, ddof=0).

BN statistics: 160 layer/checkpoint records; median mean shift 0.0144931, median variance shift 0.0658878. Layerwise absolute and relative changes are in LITEBN_PRECISEBN_LAYER_STATS.csv.

All learnable tensors and non-BN buffers were bitwise identical; maximum parameter change is zero. BN moments used all legal inner-train source examples exactly once; batch size 128, deterministic subject/session/trial order, float64 accumulation, dropout disabled.

## Interpretation

The predefined four-task improvement/variability criteria are not met. This uniform Precise-BN procedure does not justify promotion to the next LiteBN method on this diagnostic. Do not rescue individual tasks by selective application or tuning.

Current heldout is an already-open internal diagnostic, not untouched final confirmation. Historical averaging is restricted to the requested seed0 (five checkpoints), not all fifteen historical seed/fold replicates. Bootstrap resamples subjects after checkpoint averaging, 10,000 draws, seed0; no multiple-task adjustment.

Source provenance caveat: MI original training commit was not recorded; source inspection commit is labeled as such. Original selected binary hashes, source normalizer metadata and historical per-subject replay provide the recovery verification.

```text
SEED = 0
LEARNABLE_WEIGHTS_UPDATED = NO
CALIBRATION_DATA = INNER_TRAIN_ONLY
INNER_VAL_USED_FOR_CALIBRATION = NO
OUTER_USED_FOR_CALIBRATION = NO
CURRENT_HELDOUT_USED_FOR_CALIBRATION = NO
NEW_SEALED_FINAL_DATA_ACCESSED = NO
```

No new training, seed1/2 evaluation, or light-tail-risk experiment was run. Existing server tasks were not terminated.
