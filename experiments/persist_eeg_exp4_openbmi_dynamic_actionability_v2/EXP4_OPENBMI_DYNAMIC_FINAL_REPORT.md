# PERSIST-EEG Exp4 OpenBMI Dynamic Actionability V2

Terminal state: **EXP4_OPENBMI_DYNAMIC_ACTIONABILITY_NOT_SUPPORTED**

## Phase-A decision

The predeclared dynamic actionability gate failed. Dynamic trajectory features did not improve prospective held-out prediction over static features, and the negative-transfer classifier did not reach the practical AUROC target. Phase B was therefore not authorized.

- Frozen five-fold search subjects evaluated: 40
- No-adaptation BA: 0.881250
- Legal Generic trajectory BA: 0.885000
- Strongest fair Generic context BA (frozen prior audit, not used for this gate): 0.89475
- Mean Future ΔBA: +0.003750
- Dynamic RMSE reduction vs static: -0.0706
- Dynamic RMSE-improved folds: 2/5
- Dynamic Spearman: -0.017390181867817593
- Best incremental dynamic family: M_gradient
- Negative-transfer AUROC static/dynamic: 0.32352941176470584 / 0.35784313725490197
- Gradient-sign audit: PASS
- First-order utility direction agreement: 1.0

## Data access

Only V8_SEARCH was used. The 14-subject internal holdout, historical outer test, and WBCIC were not accessed. No final holdout lock was created.

## Interpretation

Within this legal cached MI-specific trajectory surrogate, dynamic update conflict is not a reliable prospective action signal. The correct terminal conclusion is `EXP4_OPENBMI_DYNAMIC_ACTIONABILITY_NOT_SUPPORTED`; no additional lambda, rank, adapter, architecture, or dataset search is justified by this experiment.

Largest limitation: The Phase-A Generic trajectory is a frozen MI-specific cached-embedding residual-head surrogate, so a failed gate does not justify claims about all raw-EEG backbones.
