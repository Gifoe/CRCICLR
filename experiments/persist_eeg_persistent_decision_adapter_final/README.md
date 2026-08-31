# PERSIST-PDA (source-only final)

This package implements an auditable cross-fitted persistent decision adapter
on the pre-existing frozen ATCNet-CleanRoom representations. The population
logits and feature representation are fixed. Historical session 1 is split by
trial order into two deterministic blocks; session 2 is metrics-only for the
source transition. Subject adapters use historical labels only, diagonal
Fisher precision pooling, and a persistent/transient decomposition.

The preregistered 12-recipe source gate selected **pda_r1_lx0.5_lp0.01** by the
minimum OpenBMI/WBCIC validation delta. It failed, so the exact terminal is
**PERSIST_PDA_SOURCE_NOT_SUPPORTED**. WBCIC S2 and EEGNeX were not opened. This is a negative
result, not a claim of future-session utility.

Run on the server with `D:\Pythonproject\.venv\Scripts\python.exe`:

```text
python code/run_source.py --datasets OpenBMI WBCIC
python code/validate.py
```

The runtime directory and all raw representations are ignored and are not
part of the Git delivery.
