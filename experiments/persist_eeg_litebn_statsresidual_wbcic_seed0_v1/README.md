# LiteBN-StatsResidual — WBCIC seed 0

This experiment tests one fixed candidate on `WBCIC_MI`, seed 0, folds 0–4. It freezes the recovered historical 64-channel LiteBN checkpoints and trains only a zero-initialized `Linear(128, 2)` residual logit head over the mean and population standard deviation of the final temporal feature map.

The user explicitly selected the recovered historical 64-channel Windows-server checkpoints after a provenance audit showed that the newer seven-backbone checkpoints have only 32 final channels. Therefore, the replayed historical B0 values in the machine-readable artifacts are authoritative for this run; the different same-Linux rounded references in the original prompt are not mixed into the comparison.

Run on the authorized Windows server with:

```powershell
E:\Anaconda\envs\persist_stable_251\python.exe -u `
  experiments\persist_eeg_litebn_statsresidual_wbcic_seed0_v1\code\run_statsresidual_wbcic.py `
  --repo D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK `
  --recovered D:\nips-temp\TotalP\P1\precisebn_recovered_historical `
  --cache D:\nips-temp\TotalP\P1\srgeo_combined_cache `
  --historical-runtime D:\nips-temp\TotalP\P1\carrier_5fold_multiseed_stability_runtime `
  --runtime D:\nips-temp\TotalP\P1\statsresidual_wbcic_seed0_runtime
```

The selected residual checkpoints remain in the external runtime directory. Repository outputs contain their hashes and full B0 checkpoint provenance without committing binary checkpoints or EEG data.

Final result: `STATSRES_NO_USEFUL_SIGNAL`.

- Outer B0 BA: `0.787040159661`
- Outer StatsResidual BA: `0.786394998371` (`-0.064516` pp)
- Internal-heldout B0 BA: `0.792200000000`
- Internal-heldout StatsResidual BA: `0.792100000000` (`-0.010000` pp)
- Selected epochs: `[0, 0, 1, 0, 0]`

The internal-heldout cohort is development/model-selection data, not a sealed independent final test.
