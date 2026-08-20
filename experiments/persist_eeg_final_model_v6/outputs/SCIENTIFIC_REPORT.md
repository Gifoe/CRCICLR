# PERSIST-EEG V6 scientific report

## Decision

Terminal state: `V6_OPENBMI_TARGET_ONLY`. All estimates are exploratory development results.

OpenBMI best legal result is `MI_SPECIFIC_BACKBONE_ADAPTED` at **83.20% BA**, or **+7.91 pp** versus the frozen EEGNet proxy (subject-bootstrap 95% CI **[+5.63, +10.31] pp**). WBCIC best legal result is `V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED` at **82.08% BA**, or **+2.65 pp** versus frozen EEGNet (subject-bootstrap 95% CI **[+1.68, +3.71] pp**).

The secondary +5 pp-over-EEGNet goal was not reached on both benchmarks. The scientifically stricter +5 pp-over-strongest-information-matched-baseline target was not reached.

## Direct answers

1. Strongest OpenBMI information-matched generic: `MI_SPECIFIC_BACKBONE_ADAPTED`, BA 0.8320.
2. Strongest WBCIC information-matched generic: `V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED`, BA 0.8208.
3. Target information: OpenBMI outcome S1 labels only; WBCIC outcome S1/S2 labels only. Future labels are scoring-only.
4. Best PERSIST-vs-generic OpenBMI delta: -0.011851851851851862.
5. Best PERSIST-vs-generic WBCIC delta: -0.0005983616654348372.
6. The primary dual matched +5 pp target: False.
7. Subject-bootstrap CIs for best-vs-EEGNet, best-vs-pre-V6 matched anchor, and PERSIST-vs-strongest-generic are stored in `final_candidate/DEVELOPMENT_RESULTS.json` and the final subject tables.
8. Fold positivity is reported in the same records.
9. Positive/nonnegative subject fractions are reported in the same records.
10. Worst-subject deltas are reported in the same records.
11. Representation adaptation did not consistently beat the strongest frozen-output/history stack on WBCIC.
12. Conditional alignment did not add reliable value.
13. Prototype adaptation did not add reliable value.
14. FBCSP/geometry did not add reliable value on OpenBMI.
15. Paired-seed Fisher protection did not improve BA over its generic control.
16. No robust real-EEG PERSIST safety increment survived paired-control repair.
17. The protected mechanism was diagonal task Fisher; it was not empirically sufficient.
18. Generic head/tail/full parameters were allowed to adapt from legal history.
19. No harmful real-EEG subspace was certified.
20. Suppression was unnecessary and remained disabled.
21. The code supports K=1 OpenBMI and K=2 WBCIC histories.
22. Generic capacity-matched adaptation explained or exceeded the PERSIST result.
23. Increment uniquely attributable to PERSIST-SA was not established.
24. Weak standalone FBCSP and failed conditional adapters were excluded from the best result.
25. No outcome future-session label was used for fitting or selection.
26. WBCIC outer was not opened, enumerated, featurized, or scored.
27. Attempted families included frozen representation heads, prototypes, FiLM/affine, bilinear adapters, FBCSP, encoder fine-tuning, Fisher protection, future-session population training, selective gates, and enlarged MI-specific backbones.
28. Redesigns followed negative matched-control results; failed families remain in outputs.
29. Dual matched +5 pp reached: False; secondary dual EEGNet +5 pp reached: False.
30. Ready for outer freeze: False.

## Reproducibility warning

An initially positive WBCIC Fisher result disappeared after generic/Fisher controls were rerun with identical minibatch-order seeds. The repaired paired result is the only result retained in final tables. This repair is material and prevents a false PERSIST attribution.
