# Frozen EEGNet recording-context representation audit, seed 0

OUTER_DEVELOPMENT was historically exposed and was evaluation-only here. CHECKPOINT_VALIDATION was inventory-only. Formal final-heldout EEG reads: 0. No neural parameter was trained or changed.

The predeclared REL_MEAN class-relation gate is algebraically impossible: (c1-r)-(c0-r)=c1-c0 within each recording. The equality was checked numerically. Thus geometry cleanliness in the same-class centroid metric cannot be substituted for Gate A.

Q1. TRAIN recording-mean shift / within-session RMS = 0.2567 (EMBEDDING).
Q2. TRAIN session-within-subject variance reduction = -0.095290; see decomposition for class/residual tradeoff.
Q3. Class-relation cosine gain = 0.000000000; session-constant subtraction cannot alter this relation. Same-class centroid cosine gain = -0.308727.
Q4. TRAIN class probe BA change = 0.009993; relative/absolute class variance ratio = 1.000000; Gate E passes.
Q5. OUTER subject-equal REL-ABS BA = +0.005599, 95% subject CI [-0.008041,0.018562].
Q6. REL-global-center BA contrast = +0.005599; feature-standardization BA = 0.738181.
Q7. Real minus wrong-reference mean BA = +0.016881; p95 fold controls and 200 draws/fold retained.
Q8. Current REL minus S1-reference BA = +0.013694.
Q9. Context class fraction versus subject gain correlation = +0.0040; matched label-assisted diagnostic gain = -0.003668; labels never form deployed references.
Q10. Secondary largest descriptive OUTER stage gain: SPATIAL (+0.010544 BA); EMBEDDING remains predeclared primary and no OUTER stage selects the claim.
Q11. TRAIN session probe BA change = +0.014555; class probe change = +0.009993.
Q12. CURRENT_SESSION_REFERENCE_REQUIRED_BUT_GAIN_SMALL; exact next action STOP_ARCHITECTURE_DIRECTION. Gate A is algebraically unattainable under REL_MEAN.

## Locked fold table

|fold|TRAIN_subjects|OUTER_subjects|selected_B|selected_lambda|absolute_BA|REL_MEAN_BA|REL_SHRINK_BA|REL_Z_BA|REL_minus_ABS_BA|global_center_BA|S1_reference_for_S2_BA|wrong_subject_p95_BA|random_reference_p95_BA|PCA_removal_BA|absolute_relation_cosine|relative_relation_cosine|relation_improvement|absolute_class_variance|relative_class_variance|absolute_session_variance|relative_session_variance|absolute_session_probe_BA|relative_session_probe_BA|absolute_subject_ID_BA|relative_subject_ID_BA|context_class_proportion|reference_norm|transform_norm|fold_interpretation|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0.0000|26.0000|8.0000|32.0000|1.0000|0.7777|0.7865|0.7865|0.7886|0.0088|0.7777|0.7683|0.7926|0.7936|0.7766|0.7614|0.7614|0.0000|6.0167|6.0167|0.8669|1.3044|0.5351|0.5662|0.3054|0.1697|0.4766|3.1984|3.1984|REL_GAIN|
|1.0000|26.0000|8.0000|64.0000|1.0000|0.7571|0.7838|0.7838|0.7523|0.0267|0.7571|0.7684|0.7699|0.7682|0.7602|0.6960|0.6960|0.0000|6.8511|6.8511|1.1930|1.3129|0.5288|0.5673|0.2222|0.1186|0.4277|2.9090|2.9090|REL_GAIN|
|2.0000|26.0000|8.0000|64.0000|1.0000|0.7120|0.6854|0.6854|0.6888|-0.0265|0.7120|0.6833|0.7086|0.7007|0.7099|0.4913|0.4913|0.0000|8.1308|8.1308|1.5722|1.5185|0.5235|0.5192|0.1891|0.0972|0.4219|2.6558|2.6558|NO_REL_GAIN|
|3.0000|26.0000|8.0000|64.0000|1.0000|0.7041|0.7087|0.7087|0.7025|0.0046|0.7041|0.6957|0.7121|0.7130|0.6998|0.4028|0.4028|0.0000|7.4546|7.4546|1.3584|1.3493|0.5545|0.5406|0.2447|0.1165|0.4414|3.0476|3.0476|REL_GAIN|
|4.0000|26.0000|8.0000|64.0000|0.7500|0.7441|0.7586|0.7559|0.7559|0.0144|0.7441|0.7387|0.7586|0.7632|0.7607|0.5350|0.5350|0.0000|8.1700|8.1700|1.4742|1.4561|0.5000|0.5214|0.2350|0.1218|0.4414|2.5233|2.5233|REL_GAIN|

## Pooled summaries

Mean Absolute BA: 0.738993; mean REL_MEAN BA: 0.744592; mean REL_SHRINK BA: 0.744049.
Pooled REL-ABS BA: 0.005599, 95% biological-subject paired CI [-0.008041, 0.018562].
Class-relation cosine gain: 0.000000000; session-variance reduction: -0.095290; class-information change (TRAIN probe BA): 0.009993.
Gates: A=FAIL, B=FAIL, C=PASS, D=PASS, E=PASS.
Primary interpretation: `CURRENT_SESSION_REFERENCE_REQUIRED_BUT_GAIN_SMALL`. Exact next action: `STOP_ARCHITECTURE_DIRECTION`.
Formal final-heldout EEG reads: 0.
Branch: `codex/persist-eeg-context-relative-representation-audit-seed0-v1`.
Final commit SHA: provided in the reviewed GitHub commit/delivery record after push; a commit cannot contain its own SHA.
