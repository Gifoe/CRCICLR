# Final-model audit

SIRE source: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_carrier_dualdataset_screen_v1/code/run_carrier_screen.py`
SIRE SHA-256: `920af131aabc272317da128f42be9961d5592619ce85ca99192029d1181f126f`
EEGNet source: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_carrier_dualdataset_screen_v1/code/eegnet_locked.py`
EEGNet SHA-256: `f7c513c3f3cd1f326a74b4e419bd693e15ee8980a7c378f1c0bee8215b8b89dd`

SIRE trainable parameters: 48,074 = 44,872 + 48*64 + 65*2.
EEGNet trainable parameters: 34,194.
SIRE remains CompactLite_BN: temporal kernels 15/63/127; 8-to-16 grouped spatial branches; width 48; refinements 15 and 31; pool 8; 512-to-64 embedding; LayerNorm(64); 2-class head.
Only the sampler differs from the frozen reference.

## Preserved channel order

`Fp1`, `Fp2`, `F3`, `F4`, `C3`, `C4`, `P3`, `P4`, `O1`, `O2`, `F7`, `F8`, `T7`, `T8`, `P7`, `P8`, `Fz`, `Cz`, `Pz`, `FC1`, `FC2`, `CP1`, `CP2`, `FC5`, `FC6`, `CP5`, `CP6`, `FT9`, `FT10`, `TP9`, `TP10`, `F1`, `F2`, `C1`, `C2`, `P1`, `P2`, `AF3`, `AF4`, `FC3`, `FC4`, `CP3`, `CP4`, `PO3`, `PO4`, `F5`, `F6`, `C5`, `C6`, `P5`, `P6`, `AF7`, `AF8`, `FT7`, `FT8`, `TP7`, `TP8`, `PO7`, `PO8`, `Fpz`, `CPz`, `POz`, `Oz`, `FCz`
