# EEGNet P task-core specificity closure

Frozen EEGNet / OpenBMI_MI / seed 0 / folds 0–4. Equal-fold descriptive means; the detailed 20,000-draw subject-bootstrap intervals, 200-permutation controls and per-fold rows are in the CSVs. All selection, matching and decoder fitting used TRAIN_GEOMETRY only. OUTER_DEVELOPMENT was evaluation-only. CHECKPOINT_VALIDATION is separately reported and historically exposed to checkpoint selection. Formal final-heldout EEG reads: **0**.

## Q1. Utility-rank persistence overlap

The rank-weighted overlap is **1.000**. Fold P ranks are [14, 13, 10, 13, 14], U/G ranks [4, 3, 4, 8, 5], P_NOT_U ranks [10, 10, 6, 5, 9], and U_NOT_P ranks [0, 0, 0, 0, 0]. All five U_NOT_P contrasts are `NOT_ESTIMABLE_DUE_TO_SUPPORT_OVERLAP`; U_ALL=G is a set identity, not evidence that persistence causally creates utility.

## Q2. Persistence alone

P_ALL held cross-subject relation cosine **0.862** and frozen cross-subject decoder BA **0.701**. Its own-rank rank/energy/covariance matched random controls and per-fold directional counts are in `P_ALL_SPECIFICITY_CONTROLS.csv` and `DECISION_SUMMARY.json`. The locked 4/5 two-metric rule is **not met**. This is an observational geometry comparison, not a causal persistence ablation.

## Q3. Utility within persistence

G held relation cosine **0.922** versus P_NOT_U **0.863**. Frozen cross-subject BA: G **0.716**, P_NOT_U **0.708**. Class variance fraction: G **0.544**, P_NOT_U **0.360**; subject fraction: G **0.213**, P_NOT_U **0.411**. The locked materiality rule requiring both +0.05 relation and +0.02 BA in at least four folds is **not met**. Rank differs; do not interpret this as an isolated intervention.

## Q4. Persistence within utility

`NOT_ESTIMABLE_DUE_TO_SUPPORT_OVERLAP`: U_NOT_P has rank zero in every fold. U_ALL and G coincide exactly. No direct claim that persistence filters transferable utility is identified.

## Q5. Protected specificity after controls

G held relation **0.922** and cross-subject BA **0.716**. Rank-random means: **0.758 / 0.648**; top PCA: **0.902 / 0.733**; energy-selected means: **0.876 / 0.694**; covariance-selected means: **0.869 / 0.693**. The native orthogonal-control 4/5 rule is **not passed**. Specifically, G exceeds top PCA in held relation on 5/5 folds but exceeds it in frozen cross-subject BA on only 0/5. Thus failing the joint rule does not imply that PCA explains every semantic observation. Matching achieved energy-ratio ranges are [0.29105290148738383, 0.46813941707303525] for energy-selected controls and [0.28643573660020294, 0.46813941707303525] for covariance-selected controls: **none of these orthogonal draws truly matched G's energy** (their best ratio is below 0.5). Their favorable/unfavorable comparisons are therefore descriptive, not a successful energy-adjusted identification. Mean overlaps with G are 0.111 and 0.111. Per-candidate spectrum discrepancy and overlap, plus 100-random means, 95th percentiles and G percentiles, are in the matching CSV and decision JSON. `SUBJECT_BOOTSTRAP_G_MINUS_RANK_RANDOM.csv` gives paired 20,000-draw biological-subject intervals for G minus the 100-draw rank-random mean. The synthetic shaped random control is invertibly rescaled, not an orthogonal erasure control, and is not used in this gate.

## Q6. Task versus context

G class and subject fractions are **0.544 / 0.213**, compared with P_ALL **0.397 / 0.354**. Task structure is present, but subject context is not absent. Labels from prior utility selection make this an operational, not independently discovered, task core.

## Q7. Model-design decision

Mechanism interpretation: **`MIXED_OR_INCONCLUSIVE`**. Exactly one next action: **`MECHANISM_NOT_IDENTIFIED_YET`**. No new architecture is trained in this branch.

## Integrity limits

The source P and U decisions were reconstructed separately from per-block TRAIN-only persistence statistics and absolute/excess 95% utility intervals; every G coordinate and fresh geometry hash matched source commit `8a8b708b30e19a20a83902d272fbe53fe8279f12`. Utility was measured on every atomic block in the source selector before conjunction with persistence. The all-zero U_NOT_P support makes direct P-versus-utility source attribution unidentified. Matched-control achieved energy ratios, spectrum discrepancies, and G overlaps must be inspected before treating the control labels as exact matches. The optional shared-layer replication was not run: the source cached geometry does not include directly reusable independent P/U per-block activation inventories at that layer. No neural parameter or BatchNorm buffer changed. Evaluation V2 used an equivalent 60-D rotated complement for C_CURRENT, which could change featurewise-standardized decoder regularization. The corrected V4 evaluation restored the source 64-D ambient residual; the V2 artifacts remain preserved on the server and are not substituted into this report. An intermediate V3 aggregation used an overstrong negative source-attribution label despite inadequate energy matching; this report corrects the label to specificity not established, and the V3 artifact remains preserved on the server. The failed duplicate scheduled-task trigger on reconstruction fold 4 occurred after a successful immutable scientific output and was blocked by the no-overwrite guard; its failure was preserved in the runtime log, not hidden.
