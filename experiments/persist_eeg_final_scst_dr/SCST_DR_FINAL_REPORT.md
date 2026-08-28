# SCST-DR final report

## Final terminal

`FINAL_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED`

The constructive method was stopped at the validated Stage-0 terminal
`TRANSPORT_OFF_MANIFOLD`.  No SCST-DR continuation model, matched continuation
baseline, future-session performance analysis, mechanism analysis, final model
lock, or sealed-outer evaluation was run.

## What the repair established

V0 showed stable residual direction but alpha=1 overshoot.  The only authorized
scientific repair prelocked one global magnitude from {0.25, 0.50} without
changing layers, folds, seed, representations, centroids, controls, probes,
manifold estimator, bootstrap, or gates.  Alpha=0.25 repaired target-subject and
class fidelity at the final embedding in all four settings.  It still produced
3NN-to-clean manifold ratios of 1.34457 (WBCIC EEGNet) and 1.36594 (WBCIC
EEGConformer), above the frozen 1.25 maximum.  Alpha=0.50 violated the manifold
gate in every setting (ratios 1.33823-1.73031 across candidate layers).

The binary off-manifold rate relative to a norm-matched random perturbation was
not the failing component.  The failure is the absolute distance from real
same-class centroid support.  Passing subject affinity and class-probe gates is
therefore insufficient to certify the counterfactual representation.

## Final-embedding frozen-gate results

| setting_id              |   alpha |   stability_effect_mean |   stability_CI_low |   subject_affinity_improvement_mean |   subject_affinity_CI_low |   class_accuracy_change |   class_logp_change |   manifold_knn_ratio_to_clean | gate_subject_fidelity   | gate_class_fidelity   | gate_manifold   | all_gates_pass   |
|:------------------------|--------:|------------------------:|-------------------:|------------------------------------:|--------------------------:|------------------------:|--------------------:|------------------------------:|:------------------------|:----------------------|:----------------|:-----------------|
| OPENBMI_MI_EEGNET       | 0.25000 |                 0.60379 |            0.53463 |                             0.12715 |                   0.11755 |                 0.00651 |             0.00708 |                       1.16425 | True                    | True                  | True            | True             |
| OPENBMI_MI_EEGCONFORMER | 0.25000 |                 0.72534 |            0.66970 |                             0.15539 |                   0.14596 |                 0.00633 |             0.01107 |                       1.15192 | True                    | True                  | True            | True             |
| WBCIC_MI_EEGNET         | 0.25000 |                 0.57082 |            0.40886 |                             0.08184 |                   0.04240 |                 0.02003 |             0.01299 |                       1.34457 | True                    | True                  | False           | False            |
| WBCIC_MI_EEGCONFORMER   | 0.25000 |                 0.62275 |            0.46114 |                             0.09641 |                   0.06448 |                 0.02303 |             0.01377 |                       1.36594 | True                    | True                  | False           | False            |
| OPENBMI_MI_EEGNET       | 0.50000 |                 0.60379 |            0.53463 |                             0.18394 |                   0.16467 |                 0.00660 |             0.00311 |                       1.37808 | True                    | True                  | False           | False            |
| OPENBMI_MI_EEGCONFORMER | 0.50000 |                 0.72534 |            0.66970 |                             0.25854 |                   0.23959 |                 0.00674 |             0.01110 |                       1.36382 | True                    | True                  | False           | False            |
| WBCIC_MI_EEGNET         | 0.50000 |                 0.57082 |            0.40886 |                             0.09278 |                   0.00344 |                 0.02422 |             0.00970 |                       1.65370 | True                    | True                  | False           | False            |
| WBCIC_MI_EEGCONFORMER   | 0.50000 |                 0.62275 |            0.46114 |                             0.11467 |                   0.03543 |                 0.02336 |             0.00772 |                       1.67549 | True                    | True                  | False           | False            |

## Required answers

1. **Legal development resources:** OpenBMI 40 development subjects and WBCIC
   41 development subjects, using only frozen model-fit/validation roles and
   legal source Sessions 1/2.
2. **Sealed resources:** untouched and unenumerated.  OpenBMI internal 14 and
   WBCIC outer 10 were not opened.
3. **Residual stability:** supported in every setting/layer with subject-level
   bootstrap lower bounds above zero.
4. **Target-subject fidelity:** supported at `final_embedding` for both reduced
   alphas in all four settings.
5. **Class preservation:** supported at `final_embedding` for both reduced
   alphas by the independent probe gates.
6. **On-manifold validity:** not supported.  Although better than the random
   control on binary outlier rate, absolute 3NN support distance failed.
7. **Selected layer:** none; no layer/global-alpha combination passed all gates.
8. **Representation drift:** not applicable because training was prohibited.
9. **Strongest matched ERM:** not run as a continuation baseline.
10. **SCST future-session BA:** not run.
11. **Paired BA delta:** not run.
12. **BA uncertainty:** not run.
13. **Settings favoring SCST:** not evaluated.
14. **SCST versus Mixup:** not evaluated as trained methods.
15. **SCST versus random perturbation:** not evaluated as trained methods.
16. **Class conditioning:** source-side class fidelity is supported; a training
    contribution is not established.
17. **Decision consistency:** not evaluated.
18. **Subject identity:** not evaluated after training.
19. **Transport decision sensitivity:** not evaluated after training.
20. **I retained / D_T down / G up:** not supported.
21. **Outer authorization:** denied by Stage 0.
22. **Outer confirmation:** not opened.
23. **Supported claim:** reduced residual transport is directionally
    subject-faithful and class-compatible on source data, but not certified as
    manifold-valid across datasets.
24. **Unsupported claim:** no generalization, decision-robustness, or
    subject-invariance conclusion is justified.
25. **Final state:** `FINAL_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED`.

## Most serious limitation

The proposed arithmetic changes the target-subject and class-probe diagnostics
in the desired direction but does not stay sufficiently close to real WBCIC
same-class representation support.  Training on those points would test an
uncertified latent augmentation, not the stated subject-transport mechanism.
