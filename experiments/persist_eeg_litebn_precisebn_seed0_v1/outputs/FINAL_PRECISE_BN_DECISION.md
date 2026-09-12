# Precise-BN source gate: blocked, not a scientific negative

**BLOCKED_SOURCE_ARCHITECTURE_MISMATCH**

The requested four-task seed0 diagnostic is **not completed**. No baseline
replay, BN recalibration, or new EEG evaluation was performed. Neither
`PRECISE_BN_SIGNAL_FOUND` nor `PRECISE_BN_NO_USEFUL_SIGNAL` is warranted.

## Direct server evidence

| Source | Strict historical CompactLite load | Meaning |
|---|---:|---|
| OpenBMI MI selected seed0 | 5/5 pass | Architecture only; replay/provenance still required |
| WBCIC MI selected seed0 | 5/5 pass | Architecture only; replay/provenance still required |
| OpenBMI ERP selected seed0 | 0/5 pass | Different architecture |
| OpenBMI SSVEP selected seed0 | 0/5 pass | Different architecture |

The canonical committed `run_carrier_screen.py` defines depth1 kernel 15,
point1 output 64, depth2 kernel 31, and embedding input 512. The seven-backbone
ERP/SSVEP implementation instead uses kernels 9/7, point1 output 32, and
embedding input 128. All ten selected ERP/SSVEP checkpoints match that smaller
architecture, not the requested exact historical CompactLite. All ten
`latest.pt` best_state tensors equal those incompatible selected tensors.
See `STRICT_ARCHITECTURE_AUDIT.csv` for actual strict-load exception details.

Canonical source last-change commit in the locally fetched GitHub history:
`40f8f04f96c1ecf0101bb81178c3c9635e306de0`.
Smaller seven-backbone source last-change commit:
`c040904cbb8d16899f97fc8a32ac75ea1962e60b`.
These are code-history references, **not proven checkpoint-creation commits**.

Recovery inventoried 7,992 binaries/archives under server `TotalP/P1` and
inspected small checkpoint state dictionaries. The 110 matching-shape
candidates belong to carrier MI selected/epoch60, SBTR, or SRGEO-EMA runs.
ERP/SSVEP matches are SRGEO-EMA candidate weights, not historical LiteBN
baseline weights, and cannot be substituted. Archive member lists and tracked
checkpoint paths were additionally checked. The ZIP checkpoint members found
belong to earlier CT/ICG experiments, not this four-task historical baseline.

Recovery limits are explicit: 1,980 initial restricted-load errors are logged;
relevant ERP/SSVEP LiteBN latest checkpoints were subsequently inspected as
trusted project artifacts. Large/unrelated checkpoints and all historical
objects inside Git bundles were not exhaustively deserialized. Therefore this
is **no verified recoverable exact source in inspected evidence**, not proof
that a valid backup cannot exist elsewhere.

## Required answers

1. BN-stat change: none performed; population moment shifts are unmeasured.
2. Outer improvement: unknown, no paired experiment.
3. Current internal heldout improvement: unknown, no data accessed this run.
4. Fold variability decrease: unknown.
5. Four-task directional consistency: unknown.
6. Worth developing as next LiteBN method: cannot determine from a failed
   prerequisite; no conclusion about the scientific hypothesis.

## Contract flags

```text
SEED = 0
LEARNABLE_WEIGHTS_UPDATED = NO
CALIBRATION_DATA = INNER_TRAIN_ONLY
CALIBRATION_EXECUTED = NO
INNER_VAL_USED_FOR_CALIBRATION = NO
OUTER_USED_FOR_CALIBRATION = NO
CURRENT_HELDOUT_USED_FOR_CALIBRATION = NO
NEW_SEALED_FINAL_DATA_ACCESSED = NO
BN_BUFFERS_UPDATED = NO
```

Header-only scientific result files deliberately contain no fabricated values.
`LITEBN_PRECISEBN_BASELINE_REPLAY.csv` preserves historical BA for traceability
but explicitly marks replay NOT RUN; it is not a successful replay table.

To unblock: provide recoverable exact historical ERP/SSVEP selected checkpoints
for all five seed0 folds with their split/normalization/selection/result
provenance. Alternatively, changing the scientific scope to the smaller
seven-backbone LiteBN requires an explicit user decision; it is not silently
authorized by this protocol. Do not retrain substitute baselines.

Existing server training PID 25168 was preserved. Artifacts were executed on
the server; only code and lightweight audit/report files are synchronized for
GitHub publication. No EEG data or checkpoint binaries are committed.
