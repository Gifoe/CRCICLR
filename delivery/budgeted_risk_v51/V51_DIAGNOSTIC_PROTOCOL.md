# V5.1 diagnostic protocol (frozen before diagnostic results)

This closed diagnostic separates raw predictor information from finite-sample correction cost. It uses only CBraMod and method-development HMC/EEGMMIDB subjects. Temporal acquisition is primary; random is descriptive. Budgets are 5/10/20/50, with 50 diagnostic only. Alpha=delta=0.10, five source seeds, five folds, 5,000 subject bootstraps with seed 20260805.

Schemes are S1 original 3-train/1-cal, S2 exact 2-train/2-cal, S3 the same exact split with a frozen positive low-capacity scale, and S4 exploratory cross-fitted pooled calibration. S4 is not an exact split-conformal primary result.

Decision thresholds, outlier criteria, hashes, aggregation, and hard-stop rules are frozen in `V51_DIAGNOSTIC_FREEZE.json`. Formal calibration, internal final, CAP, active acquisition, adaptive budgets, and the full method remain closed.

Freeze hash: `15dc72ab1f7f40967d9dd204925089ade5943ae5cbb374bd56e654d7c3f5b006`
