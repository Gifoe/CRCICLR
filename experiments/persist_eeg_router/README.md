# PERSIST-EEG Router

This directory implements the locked PERSIST-Router development protocol for
OpenBMI MI.  It starts from the frozen Signed-V3.1 Protected subspaces and the
P5.1 V2 matched continued-training control.  All Router development data are
five-fold, subject-disjoint predictions made inside each outer TRAIN split.

The implementation deliberately refuses to read OpenBMI outer-test samples.
OpenBMI development-validation is inaccessible during the audit and Router
selection phases and is evaluated only after a train-only lock file exists.
EEGMMIDB is conditional on the locked OpenBMI Router satisfying the complete
VIABLE gate.

Run the TRAIN-only feasibility phase on the experiment server:

```powershell
python experiments/persist_eeg_router/code/persist_router.py audit --device cuda
python experiments/persist_eeg_router/code/persist_router.py r0 --device cuda
python experiments/persist_eeg_router/code/persist_router.py finalize
```

The command creates the protocol audit, cross-fitted OOF caches, action
headroom tables, routeability diagnostics, random same-rank controls, and the
hard early-stop decision under `outputs/`.  Later phases are refused unless
the preceding protocol gate authorizes them.

The completed TRAIN-only result stops after R0.  Although alternative actions
have sizeable diagnostic oracle complementarity, the selected R0 has negative
Delta BA and fails to beat entropy-only and random same-rank controls.  The
frozen progression rules therefore forbid R1-R3, OpenBMI development
validation, and EEGMMIDB replication.

No target-subject centering, target labels, target adaptation, or target-batch
statistics are used at inference.  All RNG seeds are derived from SHA256.
