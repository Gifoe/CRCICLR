# LiteBN source-only Precise-BN diagnostic

**Completed seed0, four tasks x five folds. Decision: PRECISE_BN_NO_USEFUL_SIGNAL.**

Read [final report](outputs/FINAL_PRECISE_BN_DECISION.md),
[protocol](protocol/PRECISE_BN_PROTOCOL.md), and
[recovery provenance](provenance/README.md).

Direct answers to the six requested questions:

1. BN moments changed: across 160 layer/checkpoint records, median running-mean
   L2 shift was 0.0144931 and median running-variance L2 shift was 0.0658878.
   Absolute and relative per-layer shifts are reported; no weights changed.
2. Outer did not improve: 0/4 positive tasks, equal-task delta -2.041 pp.
3. Internal heldout did not improve overall: 1/4 positive tasks, equal-task
   delta -1.589 pp; OpenBMI MI tied, SSVEP +0.057 pp, ERP -0.092 pp,
   WBCIC -6.320 pp.
4. Fold variability did not decrease overall: mean SD change +1.881 pp.
   OpenBMI task SDs decreased slightly; WBCIC SD increased by 7.773 pp.
5. There is no consistent beneficial direction across four tasks. All outer
   task means declined, and heldout directions were mixed.
6. This fixed uniform Precise-BN procedure is not supported as the next LiteBN
   method. This is not proof that BN statistics are irrelevant under every
   possible procedure; no alternative was tuned or tested here.

20/20 original checkpoints passed strict loading and historical replay before
calibration: 411 subject/checkpoint/population comparisons. All 20 post-calibration
state audits passed with bitwise identical parameters and non-BN buffers.
Results use only the already-open internal heldout, not a new final test.

The first commit on this branch contains a source-gate blocker from the initial
server. The user subsequently supplied recoverable checkpoints on a second
server; final results supersede that blocker. Initial CHECKPOINT_RECOVERY and
STRICT_ARCHITECTURE_AUDIT files are retained as historical rejected-source
evidence, not the current recovered source audit. Use the current
LITEBN_PRECISEBN_SOURCE_MANIFEST.csv for the actual sources.

Server runtime: `D:/nips-temp/TotalP/P1/precisebn_seed0_v2_runtime`.
Run `code/run_on_server.ps1` in a retained terminal session to verify/resume the
same invariant-bound run. `code/verify_completion.py outputs` validates the
published cardinalities, pairing and state audits without accessing EEG.
Checkpoint binaries and EEG data are deliberately excluded from Git.
