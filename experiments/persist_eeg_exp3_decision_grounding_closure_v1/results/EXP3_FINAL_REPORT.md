# PERSIST-EEG Experiment 3 decision-grounding closure V1

This is a development-resource closure on the reused OpenBMI MI resource. It is not an untouched replication. V1/V1.1/V1.2 are preserved.

## Explicit answers

1. Existing DDA-B reproduced: **True**. Frozen finite ratio and protected/control run comparison were audited from the original DDA outputs.
2. Existing DDA-C reproduced: **True**. The source table has 215 cells and six held runs.
3. Full-representation cross-session identity measurable: **True** (6/6 competent runs under the frozen subject-shuffle null).
4. Mean primary identity evidence I: Protected=0.01517627; non-Protected cells=0.03029875.
5. Mean finite decision dependence D: Protected=0.99822301; non-Protected cells=0.24678509.
6. RMSE(M0)=0.04597840.
7. RMSE(MI)=0.04574416.
8. RMSE(MD)=0.03149284.
9. RMSE(MID)=0.03153328.
10. Adding identity to baseline: ΔRMSE(M0−MI)=0.00023424.
11. Adding decision dependence to baseline: ΔRMSE(M0−MD)=0.01479162; 95% CI=[0.010020683681228437, 0.01928289954354892]; exact run sign-flip p=0.01562500.
12. MD outperforms MI: **True**; ΔRMSE(MI−MD)=0.01428118; 95% CI=[0.009317051275099157, 0.01915517705286835]; exact p=0.01562500.
13. Runs favoring MD over MI: 6/6.
14. Run-cluster CI for MD−MI comparison: [0.009317051275099157, 0.01915517705286835].
15. Exact sign-flip p for MD−MI: 0.01562500.
16. Identity after decision (MD vs MID): ΔRMSE(MD−MID)=0.00003470; this is descriptive and is not interpreted as proof that identity contributes zero.
17. WBCIC: EXTERNAL_SUPPORT_NOT_IDENTIFIABLE because the required frozen shared identity/decision/consequence cell table is unavailable; no WBCIC outer subject was opened.
18. Outer subjects accessed: **False**.
19. Justified claim: decision dependence is more informative than subject-identity predictability for identifying task-consequential persistent structure **only to the extent supported by the frozen MD-vs-MI test (STRONG)**.
20. Not justified: identity contains zero information, task utility and identity are independent, or nonsignificant identity increments prove zero contribution.
21. Final Experiment-3 terminal state: **EXP3_DECISION_GROUNDED_IDENTITY_INSUFFICIENT**.
22. READY_FOR_EXP4_PROTECTION_FIRST: **YES**.
23. IDENTITY_INSUFFICIENCY_CLAIM: **STRONG**.

## Paper-ready Experiment 3 conclusion

Although subject-persistent structure can carry identity information, identity predictability alone does not identify which persistent variation is task-consequential. In contrast, task-protected persistent directions are coupled to the classifier decision, and decision dependence provides superior held-out prediction of intervention consequence. These results indicate that the relevant axis for invariance is decision-level utility rather than subject identifiability per se.

These findings motivate adaptation that explicitly preserves decision-grounded protected persistence rather than indiscriminately suppressing subject-predictive structure.
