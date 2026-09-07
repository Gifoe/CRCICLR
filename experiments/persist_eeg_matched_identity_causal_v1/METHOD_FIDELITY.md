# Method fidelity

The final implementation imports the upstream OpenBMI manifest/model loader
and uses the exact V3.1 checkpoint and canonical spectrum files.  The new code
adds only deterministic train-only matching, continuous suppression, identity
calibration, and paired causal statistics.  Any post-freeze bug fix is logged
in `RESULT_LEDGER.md`.
