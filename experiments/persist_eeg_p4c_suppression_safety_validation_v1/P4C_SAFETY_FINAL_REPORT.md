# P4C-Safety Final Report

Validator: **PASS**.

1. P4B terminal is `P4B_IDENTITY_RELIABILITY_CONDITION_NOT_SUPPORTED`.
2. P4B continuous MINT transport failed (MI RMSE 0.007967329 versus MINT 0.022863854; interaction sign was wrong), so it is not rewritten as success.
3. Its separately pre-specified discovery regime had pooled DeltaRegime +0.006690019, CI [+0.000983033, +0.012903471], with S1/S2/S3/S5 all positive.
4. S4/S6 were sealed before this experiment: YES.
5. Source cube hash: `41c5373bd73f327a652c3d155ffcf90642589f35e48ce0b2a47ee30307443ec0`.
6. Normalization hash: `dfcbcfcde0536e5c673637ab6b300377b4162e5205ba555c90f73274b1c6720f`.
7. E_task: `(z_D + z_C + z_O)/3`.
8. Regime: inclusive within-setting top-z_I tertile intersected with bottom/top E_task tertile.
9. Pre-outcome assignment hash: `0a6c4caf19937ee9024b852173f07dc8b151d13806e0e5c4635fbca77d98db30`.
10. S4 Low-E/High-E counts: 15/15.
11. S6 Low-E/High-E counts: 14/16.
12. Coverage gate: PASS.
13. S4 U_low: +0.000341698.
14. S4 U_high: -0.003510890.
15. S4 DeltaRegime: +0.003852588.
16. S6 U_low: -0.002102273.
17. S6 U_high: -0.006410985.
18. S6 DeltaRegime: +0.004308712.
19. Pooled U_low: -0.000838150.
20. Pooled U_high: -0.005007713.
21. Pooled DeltaRegime: +0.004169563.
22. DeltaRegime 95% CI: [-0.001973230, +0.011134038].
23. U_high 95% CI: [-0.011772474, +0.000122837].
24. Pooled High-E matched-random specificity: -0.003498474, CI [-0.010226108, +0.002738964].
25. Highest-I baseline: pooled -0.005358612, CI [-0.014615246, +0.001499452].
26. Low-E actionability: `LOW_E_SUPPRESSION_NOT_BENEFICIAL`.
27. S4/S6 same direction: Delta True; High-E harm True.
28. Safety terminal: `P4C_SAFETY_BOUNDARY_PARTIAL_SUPPORTED`.
29. Purity terminal: `P4C_SAFETY_PURITY_PASS`.
30. P4A unfinished grid remains `OPTIONAL_PARTIAL_INVARIANCE_GRID` and paused.
31. OpenBMI sealed internal holdout: `UNTOUCHED`.
32. WBCIC outer 10: `UNTOUCHED_NOT_ENUMERATED`.
33. Post-outcome threshold/model modification: NONE.
34. Scientific principle: subject identifiability alone is insufficient evidence of nuisance; task-entangled identity can act as a prospective suppression veto. Task decoupling is an empirical nuisance-admissibility condition, not a guarantee of beneficial suppression.
35. Next-stage authorization: method-level bridge `CONDITIONAL`; final new model `NOT_AUTHORIZED_AT_THIS_STAGE`.

P4B negative and P4C-Safety are not contradictory: continuous future-utility prediction remains unsupported; this experiment tests only a coarse asymmetric suppression-risk boundary.
