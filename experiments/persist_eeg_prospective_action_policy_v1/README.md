# PERSIST-EEG Prospective Action Policy V1

This experiment asks whether realised intervention consequence can be
predicted from information available before that consequence is observed.
All inputs are development-only pilot artifacts. The sealed WBCIC outer
subjects are never read, materialised, or evaluated.

The analysis deliberately keeps three incompatible decision families
separate:

1. OpenBMI sample-level `KEEP / ERASE / AMPLIFY / GEOMETRY` routing;
2. OpenBMI DDA run-fold-block suppression;
3. WBCIC development-only backbone-fold-block suppression.

Run on the server from the repository root:

```powershell
$env:PERSIST_EEG_PILOT_ROOT = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full'
& 'D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe' `
  experiments\persist_eeg_prospective_action_policy_v1\code\run_all.py
```

The command first writes and validates the provenance audit. It fails closed
if any selected artifact reports outer-test use, and only then constructs the
legal action-outcome data and runs grouped analyses.
