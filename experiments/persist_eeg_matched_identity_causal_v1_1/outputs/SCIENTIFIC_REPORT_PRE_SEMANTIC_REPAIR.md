# PERSIST-EEG Experiment 3 V1.1 scientific report

## 1. Redesign rationale

V1 did not obtain a negative causal result: its G1 failure was a train-only union-level feasibility problem (fold 2 / seed 1 had rank(P_union)=8 and only one exact-rank non-Protected supported combination). V1.1 therefore changes only the causal unit to one frozen Protected block; Protected definition, persistence definition and the estimand are unchanged.

## 2. Freeze and scope

The block-wise control rule, train persistence certification, P-anchored identity targets, symmetric identity metric, coverage rule and primary statistics were frozen before validation task outcomes. This is a development-resource closure, not an untouched or independent replication.
Frozen Protected blocks: 10; eligible blocks: 10; eligible runs: 6/6; failures: [].

## 3. Gate answers
- G0 held-out persistence: `True`; mean=0.5662742334319298, 95% CI=[0.436542084990335, 0.6985015572320464], subjects=23.
- G1 block-wise controls: `True`.
- G2 identity equivalence: `True`; ΔID_P=0.0, ΔID_N=0.0028747358091787438, difference=-0.0028747358091787438.
- G3 causal task effect: `False`.

## 4. Primary MEDIUM endpoint
H_P=0.0 (median 0.0, CI [0.0, 0.0]); H_N=0.0001092693236714974 (median 0.0, CI [-0.00035571105072463736, 0.0006358106884057962]); ΔH=-0.0001092693236714974 (median 0.0, CI [-0.0006240111714975841, 0.0003498565821256038], sign probability 0.3506, positive-subject fraction 0.43478260869565216, nonnegative-subject fraction 0.6956521739130435, worst subject -0.0033499999999999997); positive run means=1/6.

## 5. Dose and secondary controls
Dose-response: {"HIGH": {"P": {"identity_drop": 0.030222222222222195, "task_harm": 0.04799999999999996}, "N": {"identity_drop": 0.00917344276841169, "task_harm": 0.0009448802129547199}, "n_rows": 20}, "LOW": {"P": {"identity_drop": 0.0, "task_harm": 0.0}, "N": {"identity_drop": 8.666666666666e-05, "task_harm": 1.777777777777e-05}, "n_rows": 20}, "MEDIUM": {"P": {"identity_drop": 0.0, "task_harm": 0.0}, "N": {"identity_drop": 0.0022363888888888702, "task_harm": 8.999999999999e-05}, "n_rows": 20}}.
Same-rank random-control diagnostics are in `RANDOM_CONTROL_DIAGNOSTICS.csv`; Neutral-only is not a primary comparator and was not used to alter coverage or gates.

## 6. Leakage and final claim
All artifacts set `outer_test_used=false` and `outer_membership_enumerated=false`; no validation task outcome was used for matching, control eligibility, alpha, dose, metric or gate selection.
Terminal state: `PROTECTED_CAUSAL_EFFECT_NOT_SUPPORTED`.
Utility-not-identity claim: `NO`.
READY_FOR_EXPERIMENT_4: `NO`.

After matching subject-identity removal, the frozen Protected assignment does not meet the predeclared evidence threshold for greater task harm than matched Non-Protected persistence.