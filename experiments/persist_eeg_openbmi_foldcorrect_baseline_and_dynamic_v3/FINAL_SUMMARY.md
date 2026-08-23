# PERSIST-EEG protocol repair final summary

1. branch: `codex/persist-eeg-openbmi-dynamic-actionability-v2` (new experiment directory; old V2 outputs preserved)
2. fold-cache legality: PASS
3. strict-matched subjects: 40/40
4. historical inferred vanilla EEGNet: 75.297%
5. fresh vanilla EEGNet seed0: 75.725%
6. fresh vanilla EEGNet 3-seed mean ± SD: 74.967% (seed SD in `VANILLA_EEGNET_SEED_RESULTS.csv`)
7. fresh vanilla EEGNet 95% subject CI: [0.721833, 0.777169]
8. fresh vanilla EEGNet S1-only sensitivity: 71.450%
9. old invalid NoAdapt: 88.125%
10. repaired fold-correct cached NoAdapt: 82.675%
11. repaired fold-correct Generic: 83.125%
12. verified V6 strong anchor: 83.775%
13. verified strongest Generic: 83.125%
14. difference between old invalid 88.125 and repaired NoAdapt: +5.450 pp
15. difference between fresh vanilla and historical 75.297: -0.330 pp
16. explanation for any >2 pp baseline discrepancy: the fresh run is 40-subject V8_SEARCH only, uses train-only normalization and fixed StandardEEGNet with no target adaptation; the historical value is a 54-subject aggregate with unknown session/crop/epoch/metric provenance.
17. corrected Dynamic RMSE reduction vs static: -17.318%
18. corrected Dynamic Spearman: 0.2482107056312112
19. corrected M_gradient Spearman: 0.3158770724402067
20. folds dynamic improved: 2/5
21. NT AUROC static: 0.5784313725490196
22. NT AUROC dynamic: 0.6813725490196079
23. Phase-A corrected terminal state: EXP4_OPENBMI_DYNAMIC_ACTIONABILITY_NOT_SUPPORTED_FOLD_CORRECT
24. Phase B authorized? YES
25. internal holdout used? NO
26. historical outer used? NO
27. WBCIC used? NO
28. strongest currently justified paper claim: the true vanilla EEGNet and the fold-correct cached control are now separately reproducible on the authorized 40-subject OpenBMI development set; the repaired cached dynamic actionability audit has the terminal state above.
29. strongest currently unjustified claim: that the old 88.125% NoAdapt value is a valid vanilla EEGNet baseline, or that a dynamic intervention method is supported / generalizes to the sealed holdout.
