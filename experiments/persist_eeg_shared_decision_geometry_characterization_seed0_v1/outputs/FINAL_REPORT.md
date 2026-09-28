# Shared decision geometry characterization (seed 0)

Frozen-backbone diagnostic only. No neural network was trained, no model architecture was implemented, and formal final-heldout EEG reads were 0.

All basis/rank/scaler/whitening/decoder/fusion fits used TRAIN_GEOMETRY only. OUTER_DEVELOPMENT labels entered evaluation metrics and explicitly labelled evaluation-only local decision vectors. CHECKPOINT_VALIDATION was previously used for neural checkpoint selection.

Primary overall interpretation: `SHARED_DECISION_GEOMETRY_NOT_SUPPORTED`. Exact next action: `STOP_THIS_MODEL_DIRECTION`.

Common-coordinate primary analysis uses a single TRAIN-only StandardScaler per fold on every TRAIN_GEOMETRY trial embedding. A separate TRAIN-only regularized whitening sensitivity uses floor max(1e-4*largest covariance eigenvalue, 1e-8). No diagnostic trial cap is applied; the historical neural input normalizer is fitted on all TRAIN source-session trials.

The decomposition is imposed after training: learned local decision normals are described by a shared basis plus orthogonal residual. It is not an explicit factorization of the neural network.

An earlier cap-32 diagnostic preview was excluded because it did not use every TRAIN trial embedding; see `protocol/CAP32_PREVIEW_EXCLUSION.json`. Operational startup/serialization failures are disclosed in `protocol/FAILURE_DISCLOSURES.json`; failed artifacts were preserved outside Git, and no scientific fold result was silently overwritten.

## EEGNet / OpenBMI_MI

Q1: k90 mean 27.60, effective rank 16.15, E(1/2/4) 0.197/0.252/0.347; label-null E4 95th percentile 0.255. All singular curves and null draws are in DECISION_SPECTRUM*.csv. Low-rank fold gates k90/r_eff/E4 2/5/5 of 5; label False.

Q2: mean squared canonical correlation 0.413, normalized projector distance 0.766; principal angles in CROSS_SESSION_BASIS_STABILITY.csv.

Q3: strict LOSO coverage 0.530; OUTER evaluation-only decision coverage 0.497, reconstruction cosine 0.702; rank-matched random coverage p95 0.470.

Q4: frozen OUTER subject-equal shared-only BA/F1/NLL 0.742/0.740/0.537; full-linear BA 0.741.

Q5: OUTER private decision-energy fraction 0.503; private-only BA 0.669. Private structure is not called noise.

Q6: subject-equal private rescue 0.348; stacked BA 0.742, gain over shared -0.000; private-complement label True. Oracle union is nondeployable.

Coordinate sensitivity: whitened low-rank criteria k90/r_eff/E4 pass 5/5/5 of 5 folds; whitened OUTER coverage 0.565 and shared-only BA 0.740. The low-rank support label CHANGES under TRAIN-only whitening; the structural claim is coordinate-dependent.

## EEGNet / OpenBMI_ERP

Q1: k90 mean 24.00, effective rank 15.47, E(1/2/4) 0.183/0.265/0.383; label-null E4 95th percentile 0.359. All singular curves and null draws are in DECISION_SPECTRUM*.csv. Low-rank fold gates k90/r_eff/E4 0/5/5 of 5; label False.

Q2: mean squared canonical correlation 0.507, normalized projector distance 0.702; principal angles in CROSS_SESSION_BASIS_STABILITY.csv.

Q3: strict LOSO coverage 0.591; OUTER evaluation-only decision coverage 0.608, reconstruction cosine 0.778; rank-matched random coverage p95 0.418.

Q4: frozen OUTER subject-equal shared-only BA/F1/NLL 0.844/0.798/0.302; full-linear BA 0.850.

Q5: OUTER private decision-energy fraction 0.392; private-only BA 0.833. Private structure is not called noise.

Q6: subject-equal private rescue 0.251; stacked BA 0.845, gain over shared 0.001; private-complement label False. Oracle union is nondeployable.

Coordinate sensitivity: whitened low-rank criteria k90/r_eff/E4 pass 5/5/5 of 5 folds; whitened OUTER coverage 0.751 and shared-only BA 0.841. The low-rank support label CHANGES under TRAIN-only whitening; the structural claim is coordinate-dependent.

## EEGConformer / OpenBMI_MI

Q1: k90 mean 19.00, effective rank 16.73, E(1/2/4) 0.135/0.223/0.367; label-null E4 95th percentile 0.354. All singular curves and null draws are in DECISION_SPECTRUM*.csv. Low-rank fold gates k90/r_eff/E4 0/2/3 of 5; label False.

Q2: mean squared canonical correlation 0.618, normalized projector distance 0.618; principal angles in CROSS_SESSION_BASIS_STABILITY.csv.

Q3: strict LOSO coverage 0.681; OUTER evaluation-only decision coverage 0.659, reconstruction cosine 0.809; rank-matched random coverage p95 0.646.

Q4: frozen OUTER subject-equal shared-only BA/F1/NLL 0.746/0.743/0.529; full-linear BA 0.747.

Q5: OUTER private decision-energy fraction 0.341; private-only BA 0.739. Private structure is not called noise.

Q6: subject-equal private rescue 0.126; stacked BA 0.746, gain over shared -0.000; private-complement label False. Oracle union is nondeployable.

Coordinate sensitivity: whitened low-rank criteria k90/r_eff/E4 pass 5/5/5 of 5 folds; whitened OUTER coverage 0.698 and shared-only BA 0.746. The low-rank support label CHANGES under TRAIN-only whitening; the structural claim is coordinate-dependent.

## Cross-cell answers

Q7: EEGNet MI low/unseen labels False/True; EEGNet ERP False/True. See CROSS_TASK_GEOMETRY_SUMMARY.csv; raw bases are not compared across task semantics.

Q8: EEGNet MI low/unseen labels False/True; EEGConformer MI False/True. See CROSS_BACKBONE_GEOMETRY_SUMMARY.csv; bases from different representation coordinates are not directly compared.

Q9: under the predeclared common-standardized coordinates, the three cells' low-rank/unseen-subject labels are EEGNet/OpenBMI_MI: False/True; EEGNet/OpenBMI_ERP: False/True; EEGConformer/OpenBMI_MI: False/True. The low-rank gate compares TRAIN decision spectra with TRAIN label-permutation nulls; the unseen-subject gate uses frozen OUTER evaluation. Whitening sensitivity is reported separately in each cell. These are post-hoc geometric descriptions, not causal or architectural claims.

Private-complement labels are EEGNet/OpenBMI_MI: True (mean stacked BA gain -0.000); EEGNet/OpenBMI_ERP: False (mean stacked BA gain 0.001); EEGConformer/OpenBMI_MI: False (mean stacked BA gain -0.000). Rescue and NLL can satisfy the locked complement gate even when BA gain is nonpositive; do not equate the gate with a practical accuracy improvement.

Q10: `STOP_THIS_MODEL_DIRECTION`. Only a PROCEED label licenses a later model-design experiment; no new model was trained here.

Whitening and centered-W results are separate sensitivity analyses. If they conflict with common-standardized uncentered results, the primary claim is coordinate-dependent and should be narrowed.

## Primary fold table

| Cell | Dim | Train w | k50 | k90 | r_eff | E1 | E2 | E4 | Null E4 p95 | Low rank | Session R² | OUTER coverage | OUTER cosine | Random coverage p95 | Shared BA | Private BA | Full BA | Private rescue | Stacked BA | Gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EEGNet/OpenBMI_MI/f0 | 64 | 52 | 9 | 28 | 17.73 | 0.181 | 0.234 | 0.330 | 0.256 | False | 0.413 | 0.552 | 0.741 | 0.482 | 0.804 | 0.694 | 0.808 | 0.365 | 0.799 | -0.005 |
| EEGNet/OpenBMI_MI/f1 | 64 | 52 | 8 | 27 | 15.31 | 0.204 | 0.259 | 0.356 | 0.255 | True | 0.426 | 0.513 | 0.715 | 0.468 | 0.802 | 0.738 | 0.801 | 0.391 | 0.804 | 0.003 |
| EEGNet/OpenBMI_MI/f2 | 64 | 52 | 8 | 27 | 15.61 | 0.202 | 0.258 | 0.354 | 0.255 | True | 0.412 | 0.444 | 0.664 | 0.454 | 0.690 | 0.637 | 0.686 | 0.355 | 0.691 | 0.001 |
| EEGNet/OpenBMI_MI/f3 | 64 | 52 | 9 | 28 | 18.28 | 0.173 | 0.232 | 0.330 | 0.255 | False | 0.409 | 0.491 | 0.698 | 0.467 | 0.698 | 0.649 | 0.697 | 0.264 | 0.698 | -0.001 |
| EEGNet/OpenBMI_MI/f4 | 64 | 52 | 8 | 28 | 13.80 | 0.226 | 0.278 | 0.367 | 0.254 | False | 0.403 | 0.485 | 0.693 | 0.480 | 0.716 | 0.628 | 0.715 | 0.364 | 0.717 | 0.001 |
| EEGNet/OpenBMI_ERP/f0 | 64 | 52 | 7 | 24 | 17.26 | 0.151 | 0.246 | 0.372 | 0.359 | False | 0.524 | 0.631 | 0.793 | 0.410 | 0.834 | 0.828 | 0.838 | 0.268 | 0.836 | 0.002 |
| EEGNet/OpenBMI_ERP/f1 | 64 | 52 | 7 | 24 | 17.52 | 0.153 | 0.235 | 0.364 | 0.362 | False | 0.502 | 0.585 | 0.762 | 0.420 | 0.880 | 0.878 | 0.884 | 0.308 | 0.885 | 0.005 |
| EEGNet/OpenBMI_ERP/f2 | 64 | 52 | 7 | 25 | 13.09 | 0.227 | 0.296 | 0.396 | 0.347 | False | 0.489 | 0.608 | 0.777 | 0.436 | 0.786 | 0.771 | 0.793 | 0.222 | 0.786 | 0.000 |
| EEGNet/OpenBMI_ERP/f3 | 64 | 52 | 7 | 23 | 15.89 | 0.173 | 0.256 | 0.380 | 0.366 | False | 0.520 | 0.593 | 0.768 | 0.400 | 0.874 | 0.854 | 0.880 | 0.231 | 0.874 | 0.000 |
| EEGNet/OpenBMI_ERP/f4 | 64 | 52 | 6 | 24 | 13.60 | 0.213 | 0.293 | 0.405 | 0.360 | False | 0.497 | 0.621 | 0.787 | 0.423 | 0.846 | 0.835 | 0.852 | 0.226 | 0.846 | 0.000 |
| EEGConformer/OpenBMI_MI/f0 | 32 | 52 | 7 | 19 | 17.59 | 0.118 | 0.208 | 0.357 | 0.354 | False | 0.606 | 0.659 | 0.809 | 0.655 | 0.829 | 0.819 | 0.828 | 0.197 | 0.829 | 0.001 |
| EEGConformer/OpenBMI_MI/f1 | 32 | 52 | 7 | 19 | 18.44 | 0.110 | 0.189 | 0.327 | 0.357 | False | 0.649 | 0.642 | 0.799 | 0.640 | 0.784 | 0.784 | 0.787 | 0.096 | 0.784 | 0.000 |
| EEGConformer/OpenBMI_MI/f2 | 32 | 52 | 7 | 19 | 17.94 | 0.106 | 0.208 | 0.355 | 0.356 | False | 0.619 | 0.634 | 0.793 | 0.649 | 0.646 | 0.647 | 0.648 | 0.113 | 0.647 | 0.001 |
| EEGConformer/OpenBMI_MI/f3 | 32 | 52 | 6 | 19 | 14.53 | 0.175 | 0.258 | 0.404 | 0.347 | False | 0.582 | 0.685 | 0.826 | 0.643 | 0.722 | 0.710 | 0.721 | 0.142 | 0.721 | -0.001 |
| EEGConformer/OpenBMI_MI/f4 | 32 | 52 | 6 | 19 | 15.15 | 0.164 | 0.253 | 0.392 | 0.355 | False | 0.632 | 0.674 | 0.818 | 0.642 | 0.751 | 0.738 | 0.752 | 0.082 | 0.751 | -0.001 |

## Three-cell completion summary

| Cell | Dim | TRAIN w/fold | k50 | k90 | r_eff | E1 | E2 | E4 | Label-null E4 p95 | Low-rank? | Session R² | OUTER coverage | OUTER cosine | Random coverage p95 | Shared BA | Private BA | Full BA | Private rescue | Stacked BA | Stacked gain | Low-rank label | Unseen label | Private label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| EEGNet/OpenBMI_MI | 64 | 52 | 8.40 | 27.60 | 16.15 | 0.197 | 0.252 | 0.347 | 0.255 | False | 0.413 | 0.497 | 0.702 | 0.470 | 0.742 | 0.669 | 0.741 | 0.348 | 0.742 | -0.000 | False | True | True |
| EEGNet/OpenBMI_ERP | 64 | 52 | 6.80 | 24.00 | 15.47 | 0.183 | 0.265 | 0.383 | 0.359 | False | 0.507 | 0.608 | 0.778 | 0.418 | 0.844 | 0.833 | 0.850 | 0.251 | 0.845 | 0.001 | False | True | False |
| EEGConformer/OpenBMI_MI | 32 | 52 | 6.60 | 19.00 | 16.73 | 0.135 | 0.223 | 0.367 | 0.354 | False | 0.618 | 0.659 | 0.809 | 0.646 | 0.746 | 0.739 | 0.747 | 0.126 | 0.746 | -0.000 | False | True | False |

Primary overall interpretation: `SHARED_DECISION_GEOMETRY_NOT_SUPPORTED`.

Exact next action: `STOP_THIS_MODEL_DIRECTION`.

Formal final-heldout EEG reads: `0`.

