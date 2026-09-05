# Route-B randomness audit

Terminal: `CLEAN_ROUTE_B_FOUNDATION_SIGNAL_SUPPORTED`

## Direct answers

1. Old WBCIC +11--14pp order explanation fractions (fold-level): `{"ERM_ORDER_B1": 0.5794305318829672, "ERM_ORDER_B3": 1.8541848236929122, "ERM_ORDER_B4": 6.360804993958655}`.
2. Route-B B0 numerical equivalence: `PASS` (see BASELINE_NUMERICAL_EQUIVALENCE.json).
3. Clean gains are in CLEAN_EARLY_SCREEN_SUMMARY.csv; they use clean B0 and common order/RNG.
4. Clean provisional passing methods: `B2_SUBJECT_EPISODIC_MLDG`.
5. Any method exceeding the observed pure-ERM WBCIC randomness envelope: `[]`.
6. Full Route-B is not automatically resumed; manual decision is required after this audit.
7. Recommended winner by clean worst-dataset effect/consistency: `B2_SUBJECT_EPISODIC_MLDG`.
8. Multiple method order explanations >=0.5: `True`.

Route-B folds are internally prelocked but are not bitwise identical to the previous nested-OOF experiment's folds.
Canonical outcome labels, OpenBMI sealed holdout, and WBCIC outer-10 were not opened.
