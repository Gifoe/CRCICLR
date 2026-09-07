# Checkpoint provenance

- Frozen base commit: `12ab811c2a6194192b430f9c010781acd1c0379f`.
- Frozen run grid: 5 folds × 3 seeds.
- Required checkpoint entries audited: 90; all file SHA256 values matched their `RUN_LOCK.json` entries.
- Normalizers: 15/15 SHA256 matches; each contains exactly the 32 source subjects and sessions 1–2.
- PUD source checkpoint: the single pre-adaptation `PUD_SOURCE.pt` used by both A6 and A10 in final-v1.
- Physical `B0_VANILLA_EEGNET.pt` files absent: 6.
- Unambiguous B0→B1 aliases: 6; locations: fold 2/seed 0, fold 2/seed 1, fold 2/seed 2, fold 4/seed 0, fold 4/seed 1, fold 4/seed 2.

For those aliases, the frozen lock records the B0 path as `B1_STRONG_EEGNET.pt`, the selected B1 configuration is `EEGNET_F8`, B0 and B1 seeds/epochs/parameter counts match, and the hashes are identical. No file was synthesized, copied, or renamed. The complete per-run audit is in `results/checkpoint_audit.csv`.
