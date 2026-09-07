# PERSIST-EEG SCAA Reliability Stage-0.5 final report

## Frozen answers

1. Only WBCIC 41 development subjects used: **yes**.
2. Outer 10 untouched and unenumerated: **yes**.
3. All reliability features computed without S3: **yes**; feature extraction read signal sessions S1/S2 only.
4. Decision metric: `For anchor correct-class margins, within each class compute abs(mean(S1val)-mean(S2))/sqrt((var(S1val)+var(S2))/2); score is negative mean across classes, so larger is more stable.`
5. Representation metric: `Within each class compute Euclidean S1val-to-S2 final-embedding centroid distance divided by sqrt(mean of the two within-session mean squared radii); score is negative mean across classes.`
6. Adaptation-effect metric: `For the adaptation-effect scalar, within each class compute abs(mean(S1val)-mean(S2))/sqrt((var(S1val)+var(S2))/2); score is negative mean across classes.`
7. Certificate precision: `Within each seed, paired class-stratified S2 bootstrap (2000 resamples) of Delta2 BA; SE=bootstrap SD, SNR=Delta2/SE, LCB90=Delta2-1.2815515655*SE.`
8. Feature definitions changed after S3 association: **no**.
9. Best sign-persistence feature: **decision_stability**.
10. Harmful-certificate predictors: see `CROSS_VALIDATED_RELIABILITY.csv`; no outcome-driven feature selection was used.
11. Best R_sign CV AUROC: **0.5809**.
12. Subject-bootstrap 95% CI: **[0.4785, 0.6814]**.
13. Raw Delta2 R_sign AUROC: **0.5341**.
14. Identity comparison: **unavailable rather than fabricated**.
15. Backbone-only R_sign AUROC: **0.5226**.
16. Explains EEGConformer-vs-EEGNet: **False** under the frozen Gate C.
17. Backbone coefficient: **0.8034 -> 0.8220** after decision_stability.
18. Cross-backbone consistency: interaction/main absolute ratio **1.5263**.
19. Simple S2-gate coverage: **0.4756**.
20. Simple S2-gate S3 harm: **0.3077**.
21. Reliability-gated coverage: **0.2927**.
22. Reliability-gated S3 harm: **0.3333333333333333**.
23. Relative harm reduction: **-0.08333333333333322**.
24. Anchor S3 BA: **0.786958**.
25. Always-Adapt S3 BA: **0.789887**.
26. Simple S2-gated BA: **0.790209**.
27. Reliability-gated BA: **0.789254**.
28. Nontrivial adaptation rate: **True**.
29. Gates A-F: `{'A_prospective_observability': True, 'B_cross_validated_prediction': False, 'C_explains_backbone_contrast': False, 'D_harm_reduction': False, 'E_nontrivial_coverage': True, 'F_performance': True}`.
30. Authorization: **RELIABILITY_GATED_SCAA_DEVELOPMENT_NOT_AUTHORIZED**.
31. Strongest justified claim: The audit quantifies whether frozen S1/S2 stability features add out-of-subject information; it does not establish a deployable SCAA rule.
32. Stronger unsupported claim: No independent reliability confirmation, final SCAA improvement, or outer-subject generalization is established.
33. Terminal: **RELIABILITY_MECHANISM_PARTIAL**.

The subject is the statistical unit throughout: the two backbone rows are held out and bootstrapped together.
