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

## V3 seed-grouping statistical repair

- Diagnosis: pre-primary static review found that the draft D>I runner used fold-by-seed as the leave-one-run-out and bootstrap group. That permits the same fold under other random seeds to remain in the regression training data and violates the explicit rule that seeds are not independent people.
- Change: all three seeds are now held out together by fold; the ridge comparison is leave-one-fold-out; the 10,000 hierarchical bootstrap resamples folds independently within dataset and synchronizes each sampled fold across the two FMs of that dataset.
- Evidence available before change: source-validation training logs only. No held-out FM task BA, D>I consequence, S2/S3 utility, FM SCST, or sealed-resource result was generated or inspected.
- Prediction: wider and more defensible D>I uncertainty; no directional outcome prediction.
- Result: pending primary run.
- Keep/reject: KEEP and include in the pre-outcome protocol hash lock.
