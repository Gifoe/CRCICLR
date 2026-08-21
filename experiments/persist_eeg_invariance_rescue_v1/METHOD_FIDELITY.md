# Method fidelity status

Overall: `PASS`.

## A_SUBJECT_GRL_EEGNET

Status: `PASS`. Task-only calibration BA was `0.6800`; the GRL ladder at
lambda 0.05/0.10/0.30/1.00 reached `0.6583/0.6317/0.6483/0.6033`.
All models exceed the frozen invariant competence threshold 0.55 and have
identical parameter counts.

## B_EEG_DG

Status: `DEVIATION`. B0 calibration BA was `0.7217`; the binding final B1
smoke reached `0.6167`. Both exceed their competence thresholds and have
identical parameter counts. This is a clean-room method-level reproduction,
not an exact official-code replication.

## C_SCLDGN

Status: `DEVIATION`. C0/C1 calibration BA was `0.7600/0.7433`; both exceed
their competence thresholds and have identical parameter counts. This is a
clean-room method-level reproduction, not an exact official-code replication.

## Pre-freeze repair retained in the ledger

The first B1 draft aligned final mixed task features directly across subject
groups and collapsed at calibration BA 0.500 while B0 reached 0.722. A first
expert-stack repair reached BA 0.555 but its four-branch MMD remained unstable.
Both runs are excluded from science but retained on the execution server. The
final implementation rotates one two-source special-branch pair per minibatch
and uses a matched longer B0/B1 full schedule, reflecting the audited upstream
topology; objective names and weights did not change.

No outcome loader was constructed during training. Outer test used: `false`.
