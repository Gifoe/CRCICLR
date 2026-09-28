# Attempt disclosures

The first TRAIN fold-0 wrapper (`run_train_task_v1.ps1`) stopped after global-head selection without an exit marker or scientific result. Its mixed PowerShell/native-process redirection did not retain a usable traceback. The original log remains on the original server as `train_fold0_v1.log`; no OUTER data were opened.

The version-2 wrapper retained stdout/stderr and exposed a deterministic code error: at correction rank 1, scikit-learn returned a scalar prediction and the subsequent matrix multiplication raised `ValueError: matmul: Input operand 0 does not have enough dimensions`. Its logs and `train_fold0_v2.exit` remain on the server. It produced no TRAIN seal or OUTER result.

The version-3 TRAIN wrapper used immutable `run_v2.py`, which retains a one-dimensional coefficient vector at rank 1. The five fold-specific TRAIN seals were subsequently created without overwriting either failed attempt. All stage logs, runtime arrays, old code versions, and scheduled task records remain on the original server; only compact reviewed evidence is intended for Git.

Original-server failure artifact SHA256 values:

| Artifact | SHA256 |
|---|---|
| `train_fold0_v1.log` | `c5799d2527c569cde5d3e2345f9ba106bd6163fe646e0d719e5c08e55458344c` |
| `train_fold0_v2.stdout.log` | `6611c0d7e420ce5914da14b253bee984484edea3bffb94bd57a0cb7195d781c0` |
| `train_fold0_v2.stderr.log` | `553d3e65cf2e60306a90be4a0f08a772287524b666c7f3f94649339487442cd8` |
| `train_fold0_v2.exit` | `4fa791493c3299954306be6f84ce08e863f979c9d5f912bfe666b50ce01ae906` |
