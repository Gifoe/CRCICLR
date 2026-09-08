# AMSE bug-fix ledger

No outcome-driven changes.
- 2026-09-09: resume RNG tensors are loaded onto CUDA by map_location; restore_rng now converts torch and CUDA RNG states to CPU ByteTensor before restoration. This is device-compatibility only and does not alter fresh-run numerics.
- 2026-09-09: final aggregation used attribute access on summary dict (r.dataset); corrected to key access (r[dataset]). Reporting-only bug; no training or outcome logic changed.
