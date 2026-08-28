# FM iteration ledger

## V0 official-checkpoint full fine-tuning

- Diagnosis: official checkpoints and final 200-D representations load correctly; dataset adapters must repair only sampling, unit and channel-index requirements.
- Change: maximal legal channels, 200-Hz four-patch input, official checkpoint, full-model AdamW fine-tuning and a new two-class head.
- Evidence available: repository/checkpoint/input audits and source-validation only.
- Prediction: competent source validation without layer or outcome search.
- Outcome evidence inspected: NO.
- Keep/reject: pending the frozen source-validation search.

## V2 historical SCST gate-equivalence repair

- Diagnosis: a pre-primary static comparison against the hash-locked final SCST Repair-2 implementation found that the draft FM runner summarized same-subject cross-session residual cosine directly and used absolute affinity changes. The frozen historical protocol instead requires matched-minus-mismatched residual stability, relative target-affinity improvement, and SCST-minus-norm-random advantage, each aggregated at source-subject level with 10,000 subject bootstrap resamples and the applicable CI gates.
- Change: restored the historical matched-minus-mismatched stability effect; relative affinity and random-control effects; source-subject grouping across folds and seeds; independent-probe BA >= 0.55; FM task competence; stability, subject-fidelity, class-fidelity, and manifold sub-gates. Added the missing SCAA zero-harm/zero-coverage guard and fixed the report-only WBCIC scale description to x20,000 with the frozen +/-250 uV bound.
- Evidence available before change: historical SCST protocol locks and code, source-validation training logs only. No held-out FM task BA, D>I consequence, S2/S3 utility, FM SCST, or sealed-resource result was generated or inspected.
- Prediction: prevents false transport authorization and makes FM SCST numerically comparable to final Repair-2; no directional outcome prediction.
- Result: pending primary run.
- Keep/reject: KEEP and include in the pre-outcome protocol hash lock.

## V4 compact-report provenance repair

- Diagnosis: the draft unified table contained rounded placeholder specialist task BAs and the competence Markdown would otherwise remain a pre-freeze placeholder after outcome completion.
- Change: replaced placeholders with exact historical ERM/SCAA anchor means from the committed OpenBMI and WBCIC result tables, made finalization write the actual FM task BA, frozen threshold, pass/fail, and margin, and made historical utility-transferability cells follow the committed Spearman CI-lower > 0 evidence instead of a hand-coded value.
- Evidence available before change: committed historical specialist results and current source-validation logs only; no FM primary outcome was inspected.
- Prediction: reporting-only correction; no effect on any FM metric or terminal.
- Result: pending finalization.
- Keep/reject: KEEP and include in the pre-outcome protocol hash lock.

## V3 seed-grouping statistical repair

- Diagnosis: pre-primary static review found that the draft D>I runner used fold-by-seed as the leave-one-run-out and bootstrap group. That permits the same fold under other random seeds to remain in the regression training data and violates the explicit rule that seeds are not independent people.
- Change: all three seeds are now held out together by fold; the ridge comparison is leave-one-fold-out; the 10,000 hierarchical bootstrap resamples folds independently within dataset and synchronizes each sampled fold across the two FMs of that dataset.
- Evidence available before change: source-validation training logs only. No held-out FM task BA, D>I consequence, S2/S3 utility, FM SCST, or sealed-resource result was generated or inspected.
- Prediction: wider and more defensible D>I uncertainty; no directional outcome prediction.
- Result: pending primary run.
- Keep/reject: KEEP and include in the pre-outcome protocol hash lock.

## V5 SCAA grouped-evidence gate repair

- Diagnosis: pre-primary gate review found that the draft pooled SCAA sign gate checked only the 0.65 point estimate and did not enforce meaningful grouped evidence above 0.5. It also did not explicitly prevent a task-weak FM from authorizing constructive rescue.
- Change: pooled sign concordance is now formed per subject with both FM rows held together, uses 10,000 subject bootstrap resamples, and requires CI lower > 0.5 for a strong rescue. Strong rescue also requires both WBCIC FMs to pass the frozen task-competence threshold; an architecture-dependent label requires the individually positive FM to be competent.
- Evidence available before change: task prompt, frozen competence thresholds, and source-validation logs only. No held-out FM task BA, S2/S3 utility, or other primary outcome was generated or inspected.
- Prediction: prevents a correlated-row or task-weak false-positive rescue; no directional outcome prediction.
- Result: pending primary run.
- Keep/reject: KEEP and include in the pre-outcome protocol hash lock.

## V0 decision

The globally selected source-validation recipes and S1-only head recipes were retained. No layer, channel, outcome, S2 or S3 search occurred. The primary protocol is now frozen.

## V6 representation-cache serialization repair

- Diagnosis: the first primary invocation stopped before producing any task-performance, D>I, SCAA, or SCST result because pandas subject identifiers were cached as NumPy object arrays while the frozen loader correctly required `allow_pickle=False`.
- Change: normalize object-valued cache fields to fixed-width Unicode inside `save_rep`; discard only the three incomplete caches created by the failed invocation so they are deterministically regenerated. The loader remains `allow_pickle=False`.
- Evidence available before change: Python traceback, cache key/dtype inspection, and the already frozen source-validation/S1-only selections. No primary outcome file existed or was inspected.
- Prediction: identical numerical representations and labels, with subject IDs serialized safely; no metric or terminal can change except that primary computation can proceed.
- Result: pending repaired primary run.
- Keep/reject: KEEP as a pre-outcome engineering repair and refresh the protocol code hash before rerun.
