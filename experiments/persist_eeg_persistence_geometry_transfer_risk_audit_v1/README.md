# PERSIST-EEG persistence-geometry transfer-risk audit V1

This directory contains the seed-0, five-fold, EEGNet-only audit of whether
cross-session persistent subject geometry predicts pseudo-unseen discovery
subject transfer difficulty.

Run on the authorized server (runtime paths are external and ignored):

```text
E:\Anaconda\python.exe code\run_audit.py --phase preflight --device cpu
E:\Anaconda\python.exe code\run_audit.py --phase outcome --device cpu
```

`preflight` trains A-only EEGNet models, freezes descriptor support, rebuilds
train-only persistence bases, verifies the G0 geometry gate and writes the
pre-outcome lock.  `outcome` requires that lock and evaluates only discovery
query trials.  The terminal and PGEG authorization are in
`results/FINAL_DECISION.json`; the compact narrative is
`results/FINAL_REPORT.md`.

No PGEG training is started by this audit.  Runtime/checkpoint/cache files are
not deliverables.
