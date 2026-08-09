# Provenance

- Base branch: `v2-joint-risk-benefit` at `f576ea249603f112c71f3825a7eb1707e4008591`.
- V3 branch: `v3-probecert-policy-crc`.
- Baseline tests: 112 passed; final tests and coverage are stored under `outputs/v3_probecert/provenance/`.
- The initial EEGMMIDB whole-patch time shift failed source validation (26.2% agreement). Those outputs were excluded and moved to `/root/autodl-tmp/hsc_tta_eeg/outputs_invalid_v3_mi_time_shift_20260804`.
- Final EEGMMIDB time shift uses 5% interpolation and achieved 96.7% mean source argmax agreement. Action search, Oracle analysis, Probe surfaces, and all 25 EEGMMIDB nested folds were rerun after correction.
- Raw EEG, token caches, checkpoints, subject-level parquets, and outputs are excluded from Git; their paths and hashes are in `ARTIFACT_MANIFEST.json`.
