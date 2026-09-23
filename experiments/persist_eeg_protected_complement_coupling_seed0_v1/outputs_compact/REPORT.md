# Frozen Protected/complement coupling audit

At the classifier input, P plus complement reconstructs the complete representation. “Not reweight-recoverable” therefore means failure of the restricted scalar intervention family, not missing representation outside P and C.

Frozen mappings, PCA, surrogate fitting, and predictor regularization are TRAIN-only. Class-conditioned outer 2x2 pairs use true classes for retrospective mechanism evaluation. The separately provenanced label-free prediction repair uses frozen native predictions and unlabeled same-subject/session batches; outer true labels are reserved for target evaluation. This is transductive batch evaluation, not online isolated-trial forecasting. Statistical bootstrap unit: biological subject. Final-heldout access: NO.


## Prediction-feature engineering correction v2

The original outer-development error-prediction features used true class labels to select donors, so their AUROC values are invalid as prospective evidence and are preserved only in the original aggregate. The corrected analysis selects same-subject/session donors by frozen native predicted class and nearest activation norm, without true labels in feature construction. True labels enter only TRAIN targets and outer evaluation metrics. The correction was specified after the original outer outcomes had been viewed; therefore its future-error gains are exploratory and do not independently satisfy success criterion C. The original class-conditioned 2x2 mechanism and random-partition results are unchanged.

## EEGNet / OpenBMI_MI

1. Final reconstruction: max centered-logit error 2.38e-07; P+C is complete at classifier input.
2. A NOT_REWEIGHT_RECOVERABLE error does not imply a third representation block; the scalar (beta, alpha) family is restricted.
3. Layer P and C effects: see PC_MAIN_EFFECTS.csv; peak P/C interaction is temporal_bn.
4. P dependence on C: peak interaction norm 0.448951; context cosine and sign changes are in PC_INTERACTION.csv.
5. C dependence on P: the same interaction vector governs the symmetric conditional contrast, with opposite modulation sign.
6. Strongest interaction: temporal_bn.
7. Random specificity: 2 layer(s) have subject CI above zero and at least 4/5 positive folds; full random percentiles and empirical p values are in PC_RANDOM_SPECIFICITY.csv.
8. Correct/error direction: PC_ERROR_LINKAGE.csv reports the true-margin sign, correct/error strata, P reversal, and C rescue without inferring benefit from magnitude alone.
9. Corrected label-free future-session native-error AUROC: geometry BASE 0.703295, ALL_LAYER 0.704245; paired subject delta 0.000950362 (95% CI [-0.00997663, 0.0116929]). This post-outcome engineering correction is exploratory, not a pre-registered success-criterion C test.
10. Frozen-logit surrogates at depth_point_elu_pool2: additive R2 0.579081, bilinear R2 0.65434, delta 0.0752585; full layer profiles are in PC_FUNCTIONAL_SURROGATE.csv.
11. Generic capacity control: matched one-hidden-layer MLP R2 at that layer was 0.701869; its capacity was matched to the TRAIN-selected bilinear rank.
12. Random surrogate specificity: P minus the 20 equal-rank random bilinear gains was 0.0400708 (subject 95% CI [0.0141044, 0.0715403]) at that layer; all random draws appear in PC_RANDOM_SURROGATE_CONTROL.csv.
13. Cross-session stability: matched subject/class intervention-vector comparisons and their random controls are in PC_SESSION_STABILITY.csv.
14. Architecture contrast: compare this section's layer profile and summary with the other architecture; no architecture-specific mechanism is asserted from an uncorrected peak alone.

## EEGNet / OpenBMI_SSVEP

1. Final reconstruction: max centered-logit error 9.54e-07; P+C is complete at classifier input.
2. A NOT_REWEIGHT_RECOVERABLE error does not imply a third representation block; the scalar (beta, alpha) family is restricted.
3. Layer P and C effects: see PC_MAIN_EFFECTS.csv; peak P/C interaction is depth_point_elu_pool2.
4. P dependence on C: peak interaction norm 2.81131; context cosine and sign changes are in PC_INTERACTION.csv.
5. C dependence on P: the same interaction vector governs the symmetric conditional contrast, with opposite modulation sign.
6. Strongest interaction: depth_point_elu_pool2.
7. Random specificity: 3 layer(s) have subject CI above zero and at least 4/5 positive folds; full random percentiles and empirical p values are in PC_RANDOM_SPECIFICITY.csv.
8. Correct/error direction: PC_ERROR_LINKAGE.csv reports the true-margin sign, correct/error strata, P reversal, and C rescue without inferring benefit from magnitude alone.
9. Corrected label-free future-session native-error AUROC: geometry BASE 0.925928, ALL_LAYER 0.927577; paired subject delta 0.00164861 (95% CI [-0.00346693, 0.00751201]). This post-outcome engineering correction is exploratory, not a pre-registered success-criterion C test.
10. Frozen-logit surrogates at temporal_bn: additive R2 0.329097, bilinear R2 0.330615, delta 0.00151784; full layer profiles are in PC_FUNCTIONAL_SURROGATE.csv.
11. Generic capacity control: matched one-hidden-layer MLP R2 at that layer was 0.343334; its capacity was matched to the TRAIN-selected bilinear rank.
12. Random surrogate specificity: P minus the 20 equal-rank random bilinear gains was 0.00162788 (subject 95% CI [-0.000340281, 0.00429584]) at that layer; all random draws appear in PC_RANDOM_SURROGATE_CONTROL.csv.
13. Cross-session stability: matched subject/class intervention-vector comparisons and their random controls are in PC_SESSION_STABILITY.csv.
14. Architecture contrast: compare this section's layer profile and summary with the other architecture; no architecture-specific mechanism is asserted from an uncorrected peak alone.

## EEGConformer / OpenBMI_MI

1. Final reconstruction: max centered-logit error 4.77e-07; P+C is complete at classifier input.
2. A NOT_REWEIGHT_RECOVERABLE error does not imply a third representation block; the scalar (beta, alpha) family is restricted.
3. Layer P and C effects: see PC_MAIN_EFFECTS.csv; peak P/C interaction is patch_tokenizer.
4. P dependence on C: peak interaction norm 0.646216; context cosine and sign changes are in PC_INTERACTION.csv.
5. C dependence on P: the same interaction vector governs the symmetric conditional contrast, with opposite modulation sign.
6. Strongest interaction: patch_tokenizer.
7. Random specificity: 3 layer(s) have subject CI above zero and at least 4/5 positive folds; full random percentiles and empirical p values are in PC_RANDOM_SPECIFICITY.csv.
8. Correct/error direction: PC_ERROR_LINKAGE.csv reports the true-margin sign, correct/error strata, P reversal, and C rescue without inferring benefit from magnitude alone.
9. Corrected label-free future-session native-error AUROC: geometry BASE 0.699049, ALL_LAYER 0.686521; paired subject delta -0.0125273 (95% CI [-0.0240577, -0.00189741]). This post-outcome engineering correction is exploratory, not a pre-registered success-criterion C test.
10. Frozen-logit surrogates at conformer_block4: additive R2 0.928516, bilinear R2 0.954319, delta 0.0258035; full layer profiles are in PC_FUNCTIONAL_SURROGATE.csv.
11. Generic capacity control: matched one-hidden-layer MLP R2 at that layer was 0.945123; its capacity was matched to the TRAIN-selected bilinear rank.
12. Random surrogate specificity: P minus the 20 equal-rank random bilinear gains was 0.0093313 (subject 95% CI [0.00639606, 0.0125451]) at that layer; all random draws appear in PC_RANDOM_SURROGATE_CONTROL.csv.
13. Cross-session stability: matched subject/class intervention-vector comparisons and their random controls are in PC_SESSION_STABILITY.csv.
14. Architecture contrast: compare this section's layer profile and summary with the other architecture; no architecture-specific mechanism is asserted from an uncorrected peak alone.

## EEGConformer / OpenBMI_SSVEP

1. Final reconstruction: max centered-logit error 1.91e-06; P+C is complete at classifier input.
2. A NOT_REWEIGHT_RECOVERABLE error does not imply a third representation block; the scalar (beta, alpha) family is restricted.
3. Layer P and C effects: see PC_MAIN_EFFECTS.csv; peak P/C interaction is patch_tokenizer.
4. P dependence on C: peak interaction norm 3.2843; context cosine and sign changes are in PC_INTERACTION.csv.
5. C dependence on P: the same interaction vector governs the symmetric conditional contrast, with opposite modulation sign.
6. Strongest interaction: patch_tokenizer.
7. Random specificity: 5 layer(s) have subject CI above zero and at least 4/5 positive folds; full random percentiles and empirical p values are in PC_RANDOM_SPECIFICITY.csv.
8. Correct/error direction: PC_ERROR_LINKAGE.csv reports the true-margin sign, correct/error strata, P reversal, and C rescue without inferring benefit from magnitude alone.
9. Corrected label-free future-session native-error AUROC: geometry BASE 0.858971, ALL_LAYER 0.85812; paired subject delta -0.000850446 (95% CI [-0.0093538, 0.00786482]). This post-outcome engineering correction is exploratory, not a pre-registered success-criterion C test.
10. Frozen-logit surrogates at conformer_block6: additive R2 0.944627, bilinear R2 0.968081, delta 0.0234536; full layer profiles are in PC_FUNCTIONAL_SURROGATE.csv.
11. Generic capacity control: matched one-hidden-layer MLP R2 at that layer was 0.888894; its capacity was matched to the TRAIN-selected bilinear rank.
12. Random surrogate specificity: P minus the 20 equal-rank random bilinear gains was 0.00387225 (subject 95% CI [0.00266623, 0.00508682]) at that layer; all random draws appear in PC_RANDOM_SURROGATE_CONTROL.csv.
13. Cross-session stability: matched subject/class intervention-vector comparisons and their random controls are in PC_SESSION_STABILITY.csv.
14. Architecture contrast: compare this section's layer profile and summary with the other architecture; no architecture-specific mechanism is asserted from an uncorrected peak alone.


## Publication gate and retained evidence

The original label-conditioned future-error AUROC and criterion C are invalid. Corrected prediction results are post-outcome exploratory. The fixed criterion A is met in 13 model/task/layer summaries with biological-subject CI above zero and at least four positive folds; this is the mechanism-based progression gate, not an error-prediction claim. Layer-wise tests are not multiplicity-adjusted. Full pair-level and random-control files remain on the original server; their SHA256 values and row counts are in PUBLISH_VALIDATION.json. Final-heldout accessed: NO.
