# Frozen SIRE layerwise Protected-formation audit, seed 0

Canonical CompactLite SIRE-EEG, OpenBMI MI, folds 0–4. The canonical selected checkpoints, PERSIST final coordinates, normalizers, split roles and H_CONCAT/H_SHARED1 geometry were hash checked. All pathway fitting used only inner-train subject-session-class centroids. The network remained in eval mode, with no optimizer and zero parameter or BN updates. Discovery subjects were used only for evaluation. Outer-dev and final-heldout EEG reads were zero. The reused checkpoints retain the historical final-heldout diagnostic exposure disclosed in the source record.

**Interpretation:** recoverability, cross-session persistence, prediction consequences and adjacent C→P transfer are distinct measurements. No stage is declared a formation transition from linear readout alone. Pooled discovery recoverability below is biological-subject equal; its 95% CI uses 20,000 biological-subject bootstrap draws. Persistence is a matched subject×class centroid statistic averaged across five folds.

## Q1. Where does final P become recoverable?

The final embedding uses the exact canonical train-fitted PERSIST whitening/eigenvector map, making its R² of 1 definitional rather than new formation evidence. Earlier stages use the source PathFit dual ridge, with subject-grouped OOF on train centroids and subject-disjoint discovery evaluation. The OOF split refits PathFit; the canonical final-P frame remains fixed from all inner-train subjects as required by the source protocol. Branches are parallel, so their rows are not serial steps.

| Stage | Train subject-grouped OOF mean R² | Discovery subject-equal mean R² [95% CI] | Train persistence rho | Discovery persistence rho |
| --- | ---: | ---: | ---: | ---: |
| BRANCH_15 | +0.238 | +0.029 [-0.018, +0.071] | +0.717 | +0.621 |
| BRANCH_63 | +0.276 | +0.060 [+0.022, +0.094] | +0.710 | +0.599 |
| BRANCH_127 | +0.240 | +0.007 [-0.095, +0.077] | +0.686 | +0.550 |
| H_CONCAT | +0.307 | +0.120 [+0.084, +0.154] | +0.722 | +0.605 |
| H_SHARED1 | +0.431 | +0.214 [+0.176, +0.249] | +0.742 | +0.675 |
| H_SHARED2 | +0.522 | +0.314 [+0.284, +0.344] | +0.751 | +0.671 |
| EMBEDDING | +1.000 | +1.000 [+1.000, +1.000] | +0.704 | +0.567 |

Coordinate-wise R², Pearson, variance-weighted R², cosine and normalized MSE are in `LAYERWISE_P_RECOVERABILITY.csv`. Per-fold serial differences and subject-paired CIs are in `P_FORMATION_CURVE.csv`.

## Q2. Where does cross-session persistence grow?

The same final-P coordinate system is used at every layer. `rho` is the mean diagonal cross-session Pearson correlation of matched subject×class centroids; the CSV also reports normalized symmetric covariance trace, centroid cosine, within-subject distance and a between-subject reference. Discovery has few matched centroids per fold, so these are development estimates.

| Serial transition | Discovery Δrho | Positive folds |
| --- | ---: | ---: |
| H_CONCAT → H_SHARED1 | +0.070 | 4/5 |
| H_SHARED1 → H_SHARED2 | -0.004 | 1/5 |
| H_SHARED2 → EMBEDDING | -0.103 | 0/5 |

## Q3. Where does P acquire predictive consequence?

The stage-specific fitted P span was erased and the exact frozen suffix was run. Four deterministic, equal-rank orthonormal random subspaces in the same stage served as controls. Positive excess means P erasure harms BA more than random erasure. The CI is paired by biological subject.

| Stage | P erasure BA loss | Random BA loss | Excess [95% CI] | P worst-session BA loss | P NLL change |
| --- | ---: | ---: | ---: | ---: | ---: |
| H_CONCAT | +0.1243 | +0.0011 | +0.1263 [+0.0957, +0.1608] | +0.1253 | +0.2643 |
| H_SHARED1 | +0.1138 | +0.0012 | +0.1147 [+0.0919, +0.1392] | +0.1190 | +0.1631 |
| H_SHARED2 | +0.1013 | +0.0006 | +0.1052 [+0.0796, +0.1318] | +0.1103 | +0.0936 |
| EMBEDDING | +0.0067 | +0.0025 | +0.0060 [+0.0011, +0.0109] | +0.0110 | -0.0418 |

The random subspaces are drawn in the fitted P orthogonal complement and are not covariance matched to the Protected span. Excess harm therefore establishes sensitivity relative to this locked control, but may also reflect concentration of native activation energy.

`LAYERWISE_UTILITY_DECOMPOSITION.csv` additionally reports intact, complement-retained and P-path-retained native continuation. The P-path-retained probe replaces C by the train centroid and can leave the native representation manifold; its utilities are diagnostic and are not additive attribution.

## Q4. Where does C causally contribute to successor P?

Each source P, structured top-16 source C, and random source-C perturbation had identical per-trial Euclidean energy (2% of the train source RMS norm). The table reports mean successor movement divided by that input energy. Directions and scales were selected without discovery labels.

| Transition | P→P | P→C | Structured C→P | Structured C→C | Random C→P | Structured minus random C→P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H_CONCAT → H_SHARED1 | 0.756 | 0.554 | 0.070 | 0.795 | 0.026 | +0.045 |
| H_SHARED1 → H_SHARED2 | 0.702 | 0.515 | 0.071 | 0.799 | 0.031 | +0.040 |
| H_SHARED2 → EMBEDDING | 0.198 | 0.304 | 0.059 | 0.177 | 0.028 | +0.031 |

These are finite perturbations of the frozen native block. They show sensitivity, not how much a future optimizer can change the block.

## Q5. Where is P/C interaction nonlinear?

The interaction residual is `F(h+dP+dC)-F(h+dP)-F(h+dC)+F(h)`, projected into successor P and C. Each fraction divides its mean norm by the sum of the corresponding separate P- and C-perturbation response norms.

| Transition | Nonlinear successor-P fraction | Nonlinear successor-C fraction |
| --- | ---: | ---: |
| H_CONCAT → H_SHARED1 | 0.0010 | 0.0053 |
| H_SHARED1 → H_SHARED2 | 0.0013 | 0.0053 |
| H_SHARED2 → EMBEDDING | 0.0111 | 0.0131 |

## Q6. Do temporal branches contribute differently?

Only the fitted P component of one branch was erased at a time; the other two branches remained native. The final-P coordinate change and classifier effect are compared with equal-rank random erasure in that same branch.

| Branch | Final-P L2 change | Random change | Excess change (positive folds) | Excess BA harm (positive folds) | Final-P cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| BRANCH_15 | 0.5029 | 0.0290 | +0.4739 (5/5) | +0.0235 (5/5) | 0.9750 |
| BRANCH_63 | 0.5142 | 0.0261 | +0.4881 (5/5) | +0.0228 (5/5) | 0.9734 |
| BRANCH_127 | 0.4308 | 0.0231 | +0.4077 (5/5) | +0.0128 (5/5) | 0.9828 |

These branch-output interventions cannot distinguish the separate causal contribution of temporal filters from the grouped spatial filters.

## Q7. Was `depth1 + point1` alone too narrow?

`LOCAL_SCOPE_TOO_NARROW`. The locked formation rule requires at least three of four criteria: discovery recoverability grows in ≥4/5 folds; persistence grows in ≥4/5 folds; structured C→P exceeds random C→P at the pooled point estimate and in ≥4/5 folds; successor P erasure has excess BA harm at the pooled point estimate and in ≥4/5 folds.

| Transition | Δrecoverability (positive folds) | Δpersistence (positive folds) | Structured-random C→P (positive folds) | Successor P-erasure excess (positive folds) | Criteria | Active |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| H_CONCAT → H_SHARED1 | +0.094 (5/5) | +0.070 (4/5) | +0.045 (5/5) | +0.1147 (5/5) | 4/4 | True |
| H_SHARED1 → H_SHARED2 | +0.100 (5/5) | -0.004 (1/5) | +0.040 (5/5) | +0.1052 (5/5) | 3/4 | True |
| H_SHARED2 → EMBEDDING | +0.686 (5/5) | -0.103 (0/5) | +0.031 (5/5) | +0.0060 (3/5) | 2/4 | False |

`H_SHARED1 → H_SHARED2` meets 3/4 criteria even though its measured persistence does not grow; the verdict does not assume monotonic persistence. `H_SHARED2 → EMBEDDING` fails the locked rule despite the definitional final-stage recoverability increase.

## Q8. Smallest evidence-supported future trainable scope

`SCOPE_B_SHARED`. Active transitions: H_CONCAT -> H_SHARED1, H_SHARED1 -> H_SHARED2. Distinctive branch outputs: BRANCH_15, BRANCH_63, BRANCH_127. This is a diagnostic scope recommendation, not a trained-model result. Branch-output measurements cannot uniquely identify temporal versus spatial filter trainability.

The minimum scope covering the active *parameterized serial transitions* is selected. Branch-output erasure shows that frozen branch signals matter, but does not establish that updating spatial or temporal filters is necessary; it therefore does not promote the recommendation from B to C or D.
The recommendation locates the observed frozen formation path. It does not prove that training the broader block will improve accuracy or Protected persistence.

| Candidate | Trainable parameter count |
| --- | ---: |
| SCOPE_A_LOCAL | 3792 |
| SCOPE_B_SHARED | 9872 |
| SCOPE_C_SPATIAL_SHARED | 12848 |
| SCOPE_D_FULL_FEATURE | 14488 |
| SCOPE_E_FEATURE_PLUS_EMBED | 47448 |

No classifier or method was trained. Large activations were held in runtime memory; only compact CSV/JSON evidence and provenance receipts are committed.
