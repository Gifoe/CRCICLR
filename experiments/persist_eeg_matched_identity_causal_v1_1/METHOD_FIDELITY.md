# Method fidelity

The final implementation imports the upstream OpenBMI manifest/model loader
and uses the exact V3.1 checkpoint and canonical spectrum files.  V1.1 treats
each frozen Protected block as the causal unit, certifies every Non-Protected
candidate with train-only cross-session persistence, uses P-anchored identity
doses and a symmetric cross-session subject-ID metric, and aggregates controls,
blocks, runs and subjects without pseudoreplication.  Any post-freeze bug fix
is logged in `RESULT_LEDGER.md`; semantic changes invalidate the freeze.
