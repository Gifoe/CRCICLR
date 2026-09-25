# Cross-subject SIRE gradient transfer — seed 0

Diagnostic only: 5 outer folds × 5 disjoint subject meta folds, six predeclared directions, three matched parameter-space virtual steps. No training checkpoint was produced. Geometry is the exact frozen layerwise PathFit geometry. It was originally fitted on all inner-train subjects, including those serving as meta-val here; no geometry is refitted in a meta split. Discovery was read after the meta results, code, primary step and decision rules were frozen. The first discovery read stopped before any candidate evaluation because the extraction lacked frozen-reference P/C cache fields. A documented cache-only code amendment supplied those fields, and the amended code hash was locked before rerunning discovery; see `protocol/META_ANALYSIS_FREEZE_AMENDMENT.json`.

## Q1. Ordinary CE gradient
Mean train/held cosine 0.444377. Primary-step held task transfer +0.000826; beneficial sign in 76.0% of meta splits.

## Q2. P-utility gradient
Mean train/held cosine 0.445425. Source P-only CE benefit +0.000796; held benefit +0.000555; beneficial sign in 76.0%.

## Q3. Persistence gradient
Mean train/held cosine -0.045132. Source persistence gain +0.000936; held gain -0.000080; beneficial sign in 36.0%. Persistence uses explicit centered Pearson, matching the evaluation quantity.

## Q4. Gradient conflict
Mean PU/PER source cosine +0.046648; negative in 8.0% of splits. Layerwise cosines and descriptive associations are in the CSV audits.

## Q5 and Q6. Balancing and conflict projection
At the primary step, raw combined held transfer (task, PU, persistence) = (+0.000564, +0.000410, -0.000025); balanced = (+0.000583, +0.000427, -0.000001); conflict-aware = (+0.000583, +0.000427, -0.000001). All are equal-displacement directions, not trained models.

## Q7 and Q8. Transferable direction and discovery

No P-informed direction passed the locked inner-meta plus discovery criterion. At the primary virtual step, mean transfer effects were:

| Direction | Meta task | Meta P-only | Meta persistence | Discovery task | Discovery P-only | Discovery persistence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL_CE | +0.000826 | +0.000486 | +0.000096 | -0.000607 | -0.000472 | +0.000030 |
| P_UTILITY | +0.000726 | +0.000555 | +0.000079 | -0.000629 | -0.000578 | -0.000007 |
| SHARED1_PERSISTENCE | +0.000101 | +0.000051 | -0.000080 | -0.000044 | -0.000010 | +0.000389 |
| PU_PER_RAW | +0.000564 | +0.000410 | -0.000025 | -0.000394 | -0.000340 | +0.000327 |
| BALANCED | +0.000583 | +0.000427 | -0.000001 | -0.000449 | -0.000396 | +0.000267 |
| CONFLICT_AWARE | +0.000583 | +0.000427 | -0.000001 | -0.000449 | -0.000396 | +0.000267 |

Positive values mean improvement. P_UTILITY improved held meta P-only CE in 19/25 splits, but its discovery P-only effect was negative on average and positive in only 2/5 outer folds. SHARED1_PERSISTENCE improved held meta persistence in 9/25 splits; its discovery mean persistence effect was positive, driven by 4/5 folds. Inner-meta behavior therefore did not reliably predict discovery behavior. Discovery was used only for the predeclared consistency check.

The symmetric conflict projection specified here does not create a distinct direction after unit-normalizing each projected gradient and their sum: for two non-antiparallel gradients, it is algebraically collinear with BALANCED. The two directions had identical P-only transfer values and at most 6e-8 numerical differences in other continuous metrics. This audit cannot estimate an independent benefit of conflict projection under this construction.


## Decision and integrity
**PERSISTENCE_GRADIENT_SUBJECT_SPECIFIC**. Do not wrap the same P objectives in subject-level meta optimization yet; change representation or objective definition first. All 930 virtual candidate evaluations restored the exact canonical model state. No persistent parameter update, BN change, or classifier change occurred. Outer-dev EEG reads 0; final-heldout EEG reads 0. The original canonical checkpoints retain the historical final-heldout diagnostic provenance caveat. Subject-level bootstrap CIs and sign fractions are in `TRANSFER_SUMMARY.csv` and `TRANSFER_SIGN_CONSISTENCY.csv`.
