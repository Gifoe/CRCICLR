# GeoSR execution repair and RAPID_TRIAGE

The previous CUDA cache implementation compared an unindexed `cuda` device
with the resident tensor's `cuda:0` device. PyTorch considers these unequal.
Consequently every batch uploaded the entire source tensor again and rebuilt
the normalized view. An explicit CUDA index restores the existing cache.

## Source-only numerical verification

Same server (RTX 5090), source fold0, model initialization, shuffled row order,
nonuniform weights, optimizer, batch size and full epoch. Both paths were
warmed up for two batches. No outcome labels were loaded. These are individual
full-epoch measurements, not repeated-run timing confidence intervals.

|Dataset|Before sec/epoch|After sec/epoch|Speedup|GPU utilization before/after|Peak VRAM before/after|
|---|---:|---:|---:|---:|---:|
|OpenBMI|25.403|1.452|17.50x|48.47% / 83.29%|10,353 / 10,353 MiB|
|WBCIC|70.300|1.905|36.91x|48.10% / 88.78%|16,995 / 16,995 MiB|

Loss and full model-state SHA-256 were bitwise identical for both datasets.
Evidence: `DEVICE_CACHE_EQUIVALENCE.json`. Timing excludes data loading and
teacher descriptor computation; whole-pipeline speedup must be reported
separately. The earlier concurrent students took approximately 51 and 149
seconds per epoch. RAPID_TRIAGE now runs one dataset worker at a time.

## Scope and recovery

The pre-outcome amendment retains only seed0, fold0, OpenBMI and WBCIC, and
SUBJECT_BALANCED_ERM versus GEOSR. All five initial-selection teacher caches
were reused for each dataset. Student selection retains the best discovery
checkpoint and skips exact-refit under that explicit amendment.

The original `run_geosr.py` and `audit_primitives.py` stay byte-identical so
their cache fingerprints remain valid. Atomic epoch progress includes model,
optimizer, Python/NumPy/Torch/CUDA RNG, early-stopping counters and best state.
OpenBMI resumed from epoch30 and WBCIC from epoch6. Progress is retained after
completion. A regression test verifies resumed parameters, optimizer and RNG
against uninterrupted training, including interruption at the stopping boundary.

The execution optimization is hash-locked in
`RAPID_TRIAGE_EXECUTION_OPTIMIZATION_LOCK.json`; it references the original
amendment and execution lock. It does not change the scientific amendment.
The evaluator verifies both worker locks and checkpoint hashes before access.

## Full-protocol execution

`code/run_geosr_accelerated.py` forwards the original full-protocol arguments
with an explicit CUDA index. It preserves the original trainer, constants,
two-stage cross-fit, data roles, outcome handling and cache fingerprints.
It may be used only if the locked triage continuation criterion passes.

The measured student epoch improvements do not establish a complete seed0
wall time: teacher early stopping, selected exact-refit epochs, geometry
computation and loading also contribute. Do not multiply the entire old
runtime by the measured 17.5x/36.9x student speedups.

Only both datasets meeting the locked clear-positive criterion authorize
restoring the full protocol. A mixed or nonpositive result stops this run.
All triage results are directional, not a formal five-fold seed0 claim.
