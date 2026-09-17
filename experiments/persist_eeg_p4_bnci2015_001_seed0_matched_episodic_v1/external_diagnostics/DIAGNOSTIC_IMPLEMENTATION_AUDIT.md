# Diagnostic implementation audit

- PEEH source: `/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK/experiments/persist_eeg_litebn_ablation_v2_seed0/code/run_peeh_bridge.py` SHA-256 `215cc8b393cc08f21f2536d9049527156e69230a08e5261a24abd8eb2c68dcc4`.
- PSWA source: `/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK/experiments/persist_eeg_litebn_ablation_v2_seed0/code/run_pswa_bridge.py` SHA-256 `7d04ff712b1eaaa4898b8ddff43b455c8e284dc706c6bd3af7a5a3a8e42c5251`.
- The adapter invokes `run_one`, `build_spectrum`, `select_protected`, `evaluate_erasure`, `canonical_basis`, `z_coordinates`, `random_sets`, and ridge probes from those exact sources; it does not recode their rules.
- Both models expose the eval-mode 64-D tensor immediately before the classifier. Frozen checkpoint and S1-normalizer hashes are recorded per output row.
- Empty Protected gives exactly zero PEEH; PSWA does not manufacture a zero or a random control for an empty assignment. Coverage is nonempty fold/checkpoint coverage out of 5.
