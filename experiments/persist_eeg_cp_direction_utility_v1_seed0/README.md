# Frozen EEGNet C-direction utility audit

This is a development-only, frozen audit of which TRAIN complement directions help or harm the downstream Protected pathway. It covers EEGNet seed 0, folds 0–4, on OpenBMI MI/ERP/SSVEP and WBCIC MI. It trains no model component and does not load `outer_dev_subjects` or formal final-heldout EEG.

The runner reuses the hash-verified discovery projectors and canonical checkpoints from `persist_eeg_native_cp_transfer_gate_v2_seed0`. It fits an unsupervised PCA basis on TRAIN spatial complement activations, freezes biological-subject bootstrap candidate labels, and only then loads `inner_val_subjects` for direction validation. Twenty deterministic equal-rank random complement bases receive the same primary native-erasure utility audit.

Run `code/run.py preflight` once. Then run all 20 `prepare` cells with `code/run_queue.py prepare --workers 4`; freeze candidate labels with `code/run.py lock-discovery`; run all 20 discovery cells with `code/run_queue.py discovery --workers 4`; finish with `code/run.py aggregate`. Set `NATIVE_GATE_V2_RUNTIME` and `CP_DIRECTION_RUNTIME` to the matching frozen V2 projector runtime and a writable experiment runtime before running.

The protocol and global discovery locks are written under `protocol/`. Required aggregated tables and the report are written under `outputs/`. Per-cell artifacts and completion hash manifests live in the runtime directory.
