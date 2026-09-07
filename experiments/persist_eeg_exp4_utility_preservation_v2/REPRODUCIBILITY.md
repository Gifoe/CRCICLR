# Reproducibility

Run `python experiments/persist_eeg_exp4_utility_preservation_v2/code/run_exp4_v2.py audit`, then `prepare`, `discover`, `generic`, and `dev`. The runner records source/cache/checkpoint hashes, deterministic fold seeds, all subject-level tables, and outer-sealed locks. Raw EEG and large caches are excluded from Git.
