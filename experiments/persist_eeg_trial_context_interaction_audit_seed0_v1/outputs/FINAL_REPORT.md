# Trial-context interaction qualification — frozen EEGNet / OpenBMI_MI / seed 0

Primary stage: EMBEDDING (64D); OUTER_DEVELOPMENT was historically exposed and is evaluation-only here. Formal final-heldout EEG reads: **0**. All B, C, K and family choices were frozen using TRAIN_GEOMETRY before new OUTER interaction evaluation. The low-rank interaction uses a supervised TRAIN-only moment-SVD factorization and a regularized linear readout; it is not a jointly optimized neural bilinear layer.

## Predeclared answers

Q1. Additive current context minus trial-only pooled BA: -0.0130. This is a descriptive OUTER contrast, not a selection criterion.
Q2. Selected interaction minus trial-only pooled BA: -0.0005.
Q3. Selected interaction minus h-r pooled BA: -0.0043.
Q4. Selected interaction minus [h,r,h-r] pooled BA: +0.0037; primary best-noninteraction contrast -0.0016, subject-cluster 95% CI [-0.0154, +0.0125].
Q5. Interaction-term ablation gain: -0.0008; Gate B FAIL.
Q6. Versus 100 matched random linear expansions: +0.0028; versus H-only expanded: +0.0057; Gate D PASS. Random linear expansion changes L2 geometry but does not add multiplicative coupling.
Q7. True minus wrong-subject current-context BA: +0.0047, subject-cluster 95% CI [-0.0039, +0.0132]; Gate C FAIL. Wrong-context mean prediction-flip fraction 0.0498, mean absolute logit change 0.4295, wrong-minus-true NLL +0.0239. TRAIN true-pairing minus 200 shuffled-pairing null mean: +0.0105.
Q8. Current S2 minus historical S1 context BA: +0.0048; Gate E PASS; strong >=.005: NO.
Q9. Absolute first-B class-imbalance vs interaction benefit correlation (descriptive): +0.0298; first-B class-1 fraction vs raw context norm correlation: +0.4522. Label-assisted balanced first-B diagnostic mean delta: +0.0010 (available n=40); this diagnostic is not deployable, and no target label enters the primary reference.
Q10. Across fold-subject appearances, improved 17/40, harmed 16/40, tied 7; bootstrap clusters repeated appearances by biological subject ID.
Q11. Low-rank BA 0.7467 at selected ranks [1, 4, 8, 2, 1], elementwise BA 0.7383, full simple BA 0.7412. Mean norm of between-class q separation: current 2.0917, historical 2.1553, wrong-reference mean 0.7956; this is a coordinate-level descriptive diagnostic, not predictive or biological evidence. Ranks/regularization were TRAIN-selected, not OUTER-tuned.
Q12. Build a plug-and-play block? NO. Exact next action: `STOP_CONTEXT_INTERACTION_DIRECTION`.

## Decision gates and interpretation

Pooled trial-only BA 0.7459; TRAIN-selected best noninteraction BA 0.7471; TRAIN-selected best interaction BA 0.7454. Primary ΔBA -0.0016 with subject-cluster 95% CI [-0.0154, +0.0125]. Interaction ablation gain -0.0008; wrong-context penalty +0.0047; historical-context penalty +0.0048; matched-capacity random/H-only contrasts +0.0028/+0.0057.

Case: `NO_TRIAL_CONTEXT_INTERACTION_ADVANTAGE`. Gates A/B/C/D/E: FAIL/FAIL/FAIL/PASS/PASS. A requires >=.010 BA, >=4/5 positive folds, and lower CI>0. B requires >=.005 BA and >=4/5 positive folds. C requires >=.010 and lower CI>0. D requires positive matched-control contrasts. E requires current >= historical.

No causal, biological, or architecture-performance claim is made from failed gates. Neither a positive single fold nor a target-label-balanced context is treated as deployable support. This audit does not evaluate formal final-heldout EEG.

Code and compact artifacts are on `codex/persist-eeg-trial-context-interaction-audit-seed0-v1`; the immutable pushed commit SHA is provided in the delivery response because a Git commit cannot contain its own hash.

## Final fold table

| fold | train_subject_count | outer_subject_count | B | TRIAL_ONLY_BA | RELATIVE_SUBTRACTION_BA | ADDITIVE_CONTEXT_BA | ADDITIVE_RELATIVE_BA | ELEMENTWISE_INTERACTION_BA | LOWRANK_BILINEAR_BA | selected_bilinear_rank | best_noninteraction | best_noninteraction_BA | best_interaction | best_interaction_BA | interaction_minus_noninteraction | interaction_ablation_BA | interaction_term_gain | wrong_mean_BA | wrong_p95_BA | current_minus_wrong | historical_BA | current_minus_historical | random_matched_BA | h_only_expanded_BA | context_only_BA | improved_subject_fraction | harmed_subject_fraction | fold_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 26 | 8 | 32 | 0.7811 | 0.7978 | 0.7867 | 0.7887 | 0.7847 | 0.7831 | 1 | RELATIVE_SUBTRACTION | 0.7978 | FULL_SIMPLE_INTERACTION | 0.7887 | -0.0091 | 0.7887 | 0.0 | 0.7821 | 0.7925 | 0.0066 | 0.7768 | 0.0119 | 0.7801 | 0.776 | 0.5 | 0.5 | 0.5 | negative_descriptive |
| 1 | 26 | 8 | 64 | 0.7713 | 0.79 | 0.7681 | 0.7763 | 0.7695 | 0.7813 | 4 | ADDITIVE_RELATIVE | 0.7763 | ELEMENTWISE_INTERACTION | 0.7695 | -0.0068 | 0.7695 | 0.0 | 0.7555 | 0.773 | 0.0139 | 0.7564 | 0.013 | 0.7663 | 0.7475 | 0.5 | 0.375 | 0.5 | negative_descriptive |
| 2 | 26 | 8 | 64 | 0.7289 | 0.6854 | 0.667 | 0.686 | 0.6969 | 0.7226 | 8 | ADDITIVE_RELATIVE | 0.686 | LOWRANK_BILINEAR | 0.7226 | 0.0366 | 0.7281 | -0.0054 | 0.716 | 0.7316 | 0.0066 | 0.7308 | -0.0082 | 0.7189 | 0.7289 | 0.5 | 0.875 | 0.125 | positive_descriptive |
| 3 | 26 | 8 | 64 | 0.6962 | 0.7118 | 0.6989 | 0.7065 | 0.6989 | 0.7002 | 2 | RELATIVE_SUBTRACTION | 0.7118 | LOWRANK_BILINEAR | 0.7002 | -0.0116 | 0.6989 | 0.0013 | 0.7032 | 0.7168 | -0.003 | 0.6947 | 0.0054 | 0.6985 | 0.6925 | 0.5 | 0.375 | 0.5 | negative_descriptive |
| 4 | 26 | 8 | 64 | 0.7523 | 0.7634 | 0.7441 | 0.751 | 0.7414 | 0.7462 | 1 | RELATIVE_SUBTRACTION | 0.7634 | FULL_SIMPLE_INTERACTION | 0.7462 | -0.0171 | 0.7462 | 0.0 | 0.7466 | 0.7544 | -0.0004 | 0.7441 | 0.0021 | 0.7495 | 0.7538 | 0.5 | 0.0 | 0.375 | negative_descriptive |

Pooled primary interaction−best-noninteraction BA -0.0016, 95% CI [-0.0154, +0.0125]; exact next action `STOP_CONTEXT_INTERACTION_DIRECTION`; formal final-heldout EEG reads 0.
