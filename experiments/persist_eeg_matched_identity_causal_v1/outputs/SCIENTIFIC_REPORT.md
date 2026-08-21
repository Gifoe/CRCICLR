# PERSIST-EEG Experiment 3 scientific report

## Scope

This prospectively frozen closure stopped before causal validation outcomes because the train-only matched-control gate was impossible for one frozen assignment. It is not an untouched independent replication and it is not outer validation.

## Gate results

- G0 held-out persistence replication: `True`; R_persist mean=0.5662742334319298, 95% CI=[0.4345339073991889, 0.7036389605180138].
- G1 matched controls: `False`; required at least 20 controls for every assignment. Train-only failed runs: [{"accepted_controls": 1, "candidate_count": 1, "fold": 2, "outer_membership_enumerated": false, "outer_test_used": false, "pool_size": 8, "protected_block_ids": [5, 6], "protected_rank": 8, "reason": "frozen non-Protected persistence-supported pool has fewer than the required 20 exact-rank controls", "required_controls": 20, "seed": 1}].
- G2 identity-dose equivalence: not evaluated because G1 failed before validation outcome.
- G3 primary causal consequence: not evaluated because no valid six-run matched-control design was frozen.

## Scientific interpretation

The natural non-Protected persistence-supported pool cannot supply the required exact-rank control ensemble for fold-2/seed-1 (Protected rank 8, non-Protected pool rank 8, one legal combination). Creating duplicate rotations of the same full-rank span would not be independent controls, and admitting unsupported coordinates would change the frozen control source; neither was used.

No H_P, H_N, Delta_H, identity-dose equivalence, or dose-response causal quantity is reported as measured. The terminal state is `MATCHED_NONPROTECTED_CONTROL_UNAVAILABLE`.

Utility-not-identity claim: `PARTIAL`. Experiment-4 entry: `NO`.

## Leakage and outer audit

Validation representations were used only for the held-out persistence audit after the train-only design failed. No validation task BA, identity-dose outcome, or task consequence was used to select matching. All artifacts set `outer_test_used=false` and `outer_membership_enumerated=false`.
