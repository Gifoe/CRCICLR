# Reproducibility

## Server

* Host: `100.94.171.11`, user `fyl412`.
* Python: `D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe`.
* WBCIC root: `D:\nips-temp\TotalP\P1\CRCICLR_WBCIC_EEGNET`.
* Raw root used only as a locked configuration: `D:\nips-temp\TotalP\P2\nm000348_v1.0.4_bids`.
* Experiment root: `D:\nips-temp\TotalP\P1\CRCICLR_EXP4_PROTECTION_FIRST_FINAL`.
* GPU: RTX 5090.

Development commands (with the environment variables set to the paths above) were:

```powershell
python run_exp4.py audit
python run_exp4.py prepare --device cuda
python run_exp4.py select_generic --device cuda
python run_exp4.py compute --device cuda
python run_exp4.py analyze --device cuda
```

V2 reused the frozen V1 anchors/bases and ran in a separate root with `PERSIST_EXP4_IMPLEMENTATION_ID=persist_eeg_exp4_protection_first_v2_decision_response`. V3 used a separate root and additionally set `PERSIST_EXP4_RESPONSE_STRENGTH=0.25`. These roots are retained on the server as compact iteration outputs; no raw EEG was copied.

## Code and provenance

* V1 code commit: `7735c37c3ea859a9cab886510ea1bcb56cb1acc2`.
* V2 code commit: `7ca8c26c036b5aec6ebc6eeea8987d2aad07af3b`.
* V3 code commit: `68d2a78833cc585fb4bfa0793c6fea321ab78006`.
* Frozen source lock hashes are recorded in `protocol/PROVENANCE_AUDIT.json`.
* Subject-level tables are in `results/DEV_SUBJECT_RESULTS.csv`; raw deterministic RandomGuard draws are retained separately.

The final protocol file is explicitly non-authorizing because the development gate failed. There is no outer-result file by design.
