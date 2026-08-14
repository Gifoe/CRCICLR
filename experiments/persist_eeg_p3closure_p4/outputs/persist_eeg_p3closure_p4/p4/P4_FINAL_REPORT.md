# PERSIST-EEG P4 Final Report

Decision: `P4_MAIN_METHOD_NOT_SUPPORTED`

Development decision: `P4_MAIN_METHOD_NOT_YET_SUPPORTED`

No method was locked. OpenBMI outer-test, the five-seed formal evaluation, persistence-risk curves, and EEGMMIDB were **not run**.

## A. Did PERSIST learn a real persistence subspace?

Not stably. V0 had zP/hF macro AUROC 0.8017/0.7348 (gap 0.0669), but failed task performance and MI-use gates. Gaps for V1/V2/V3 were 0.0465, 0.0249, and 0.0173. The semantic ordering did not survive the changes that forced task use of the persistent branch.

## B. Did the orthogonal decomposition behave correctly?

Yes as an engineering invariant. Every version passed orthogonality, reconstruction, finiteness, and rank checks. This is insufficient for scientific success because exact geometry alone does not establish persistence semantics or utility.

## C. Did tasks learn different persistence budgets?

Numerically yes, scientifically no. Mean gates (MI/ERP/SSVEP) were:

- V0: 0.168/0.217/0.314
- V1: 0.398/0.378/0.489
- V2: 0.445/0.400/0.458
- V3: 0.303/0.389/0.487

SSVEP was largest in every version although P2 found MI had the strongest erasure utility. No ordering was hard-coded. The discrepancy must be reported, not hidden.

## D. Did PERSIST preserve or improve decoding?

No version met the development warning that no task be more than 1pp below the historical EEGNet validation reference. V3 improved MI by 1.33pp and ERP by 0.29pp, but reduced SSVEP by -3.50pp.

These are development-only historical-reference comparisons, **NOT A FORMAL BASELINE COMPARISON**.

## E. What modifications were necessary?

- V0→V1: task warmup, auxiliary ramps, delayed/weaker budget.
- V1→V2: residual readout `C_F(hF)+C_P(g⊙zP)` to prevent persistent-branch bypass.
- V2→V3: rank 8→4, stronger/longer persistence ordering, weaker budget.

All modifications used TRAIN/VALIDATION only and are recorded in `P4_ADAPTATION_LOG.json`.

## F. Was any test-driven tuning used?

`NO`. Outer-test was never accessed. Because no V0–V3 candidate passed all gates, creating `P4_LOCKED_METHOD.json` or running the formal test would violate the protocol.

## Failure interpretation

The exact projector is numerically sound, but this implementation does not jointly deliver stable persistence semantics, interpretable minimum-sufficient budgets, and non-catastrophic task performance. The current method is not ready for formal baselines, ablations, external replication, or a paper claim.
