# PERSIST-EEG Decision Dependence Audit V1

This experiment tests whether frozen, subject-persistent representation
directions enter the classifier decision mechanism.  It is an audit, not a
new model search.  OpenBMI outer-test data are never loaded.

The confirmatory scope is OpenBMI MI because PERSIST-CF and the frozen P5.1
V2 matched classifier are MI experiments.  Signed-V3.1 assignments are read
without modification.

Run on the experiment server from the repository root:

```powershell
$env:PYTHONPATH = "$PWD\src"
$python = 'D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe'
& $python -u experiments\persist_eeg_dda_v1\code\persist_dda_v1.py audit --device cuda
& $python -u experiments\persist_eeg_dda_v1\code\persist_dda_v1.py dda-a --device cuda
& $python -u experiments\persist_eeg_dda_v1\code\persist_dda_v1.py dda-bc --device cuda
& $python -u experiments\persist_eeg_dda_v1\code\persist_dda_v1.py finalize --device cuda
```

`audit` must run first.  It writes the provenance audit and immutable protocol
lock before any DDA result is computed.  `finalize` refuses to authorize AGDI
unless DDA-A, DDA-B, and DDA-C all pass their frozen gates.

## Completed result

Terminal state: `DDA_PARTIAL_MECHANISM_ONLY`.

- DDA-A: `DDA_A_FAIL`. CF offsets move q by 0.228 of the TRAIN norm, but
  flip-rate/TV equivalence fails and centered-logit movement is indistinguishable
  from exact matched random offsets.
- DDA-B: `DDA_B_PASS`. Frozen PROTECTED blocks are locally and finitely more
  decision-active than matched controls in the run-level analysis.
- DDA-C: `DDA_C_PASS`. Adding finite decision dependence reduces leave-one-run-
  out RMSE by 31.5% (run-cluster 95% CI 22.8% to 41.0%; 6/6 runs; permutation
  p=0.00020).

The chain is incomplete, so external actionability work and AGDI are not
authorized: `STOP_AGDI_DDA_CHAIN_INCOMPLETE`.  Outer test remains locked.

The inherited loader materializes an all-subject manifest/h0 container but
only outer-TRAIN positions enter any computation.  See
`outputs/protocol/PROVENANCE_SCOPE_CORRECTION.json` for the exact scope caveat.

Audit the downloaded outputs without EEG data:

```powershell
python experiments/persist_eeg_dda_v1/code/audit_outputs.py
```
