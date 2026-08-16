# PERSIST-EEG Decision Dependence Audit V1

**Terminal state: `DDA_PARTIAL_MECHANISM_ONLY`**

## Executive decision

- DDA-A: `DDA_A_FAIL`
- DDA-B: `DDA_B_PASS`
- DDA-C: `DDA_C_PASS`
- AGDI training authorized: `false`
- Outer test: `OUTER_TEST_LOCKED`
- Stop state: `STOP_AGDI_DDA_CHAIN_INCOMPLETE`

The confirmatory audit used OpenBMI MI outer-TRAIN subjects only.  Signed-V3.1
assignments and the P5.1 V2 matched classifier choice were frozen before this
audit and were not redefined.

## Provenance

The machine-readable provenance map is `protocol/PROVENANCE_AUDIT.json`; the
pre-result gate freeze is `protocol/DDA_PROTOCOL_LOCK.json`.  No backbone was
retrained.  The inherited loader materialized the all-subject manifest and h0
container, but no development-validation or outer-test position, label, or
representation was indexed for fitting, measurement, statistics, or gates.
`protocol/PROVENANCE_SCOPE_CORRECTION.json` preserves this distinction and
corrects the original audit's overly broad `loaded=false` wording.

## DDA-A — CF behavioral-null explanation

Status: `DDA_A_FAIL`.  Relative q movement mean:
0.227604.  Centered-logit RMS divided
by the TRAIN margin mean: 0.055001.
Flip rate mean: 0.017725.  Formal equivalence
uses frozen one-sided bounds; a nonsignificant difference is never treated as
evidence of null.

## DDA-B — Protected decision activity

Status: `DDA_B_PASS`.  Protected/random Jacobian ratio mean:
3.445844; finite-logit ratio mean:
1.679936.  Protected exceeded matched non-PROTECTED blocks
in 5/6 runs, with signed-utility and
held-consequence concordance in 6/6.

## DDA-C — incremental held-out explanation

Status: `DDA_C_PASS`.  Baseline LORO RMSE: 0.04597840; full
RMSE: 0.03149284; relative improvement:
31.5051%; improved runs: 6/6;
permutation p=0.000200.  Decision subjects and intervention-
outcome subjects are disjoint within each audit fold.  Trial-level
pseudo-replication is not used.

## Statistical uncertainty and limitations

Inference aggregates trials to subjects before gate evaluation.  DDA-C uses
run/block cells with run-level held prediction and run-cluster bootstrap.  Six
runs limit precision.  Canonical discovery and Signed assignment used the
full outer-TRAIN split in the earlier frozen phase; DDA cross-fitting separates
classifier fitting, decision measurement, and consequence measurement, but it
does not claim an independently rediscovered basis.  The confirmatory scope is
MI because PERSIST-CF and the P5.1 V2 matched classifier are MI-specific.  The
all-row container behavior means the run satisfies a no-use lock, not a literal
never-materialize-metadata lock; this is a provenance limitation, not evidence
that a held-out row influenced any computation.

## Exact decision

`DDA_PARTIAL_MECHANISM_ONLY`

`STOP_AGDI_DDA_CHAIN_INCOMPLETE`
