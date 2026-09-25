# Stagewise Protected utility shared refinement — seed 0

## Scope and endpoint
Canonical SIRE-EEG / CompactLite, OpenBMI MI, seed 0, folds 0–4, subject-disjoint discovery. Both arms train exactly depth1/point1/depth2/point2 weights from the same canonical checkpoint for 20 AdamW epochs. All other states, including BatchNorm buffers and classifier, are frozen. The model stays in eval mode. No discovery-guided selection. The historical checkpoint provenance caveat remains: The canonical selected checkpoints have historical diagnostic evaluations on final-heldout subjects recorded in the source manifest. This experiment performs zero final-heldout EEG array reads and does not erase that historical exposure.

Each deterministic optimizer batch contains two trials from each session for each subject-class unit. All trials appear once per epoch in both arms. Student and frozen reference persistence centroids use the same sampled trials. Batch-specific frozen C0 sets the growth-only target; full inner-train centroids provide epoch diagnostics. The two arms have identical batches, exposures and optimizer steps.

## Q1 and Q4: final native task utility
Biological-subject-equal BA: baseline 0.773750, Shared-CE 0.761583, stagewise 0.759500. Stagewise minus Shared-CE -0.002083; stagewise minus baseline -0.014250. Fold deltas versus baseline: [-0.023333333333333428, -0.010833333333333472, -0.012500000000000178, -0.006666666666666821, -0.019166666666666665]. Paired 20,000-draw subject bootstrap intervals are in `PAIRED_CONTRASTS.csv`.

## Q2: Shared1 persistence
Discovery frozen 0.675310; Shared-CE 0.663044; stagewise 0.668260. This is computed using the prior layerwise fixed recoverability pathway, with no refit on discovery. Train-only Q-coordinate correlations are also in `TRAINING_HISTORY.csv`.

## Q3 and Q5: Shared2 Protected utility and attribution
Frozen BASE BA 0.769667; stagewise P-only transplant BA 0.760500. Full trial-weighted BA change -0.014500; P-only change -0.009167. P-only gain retention: undefined (defined only when both changes are positive). Shared2 final-P recoverability R2 changes from 0.362833 to 0.350385; erasure BA consequence changes from 0.101333 to 0.093167. Full mechanism results are in `MECHANISM_CHANGE.csv`. Hybrid uses student P with frozen native C and the original frozen suffix.

## Q6: prediction transitions
Stagewise wrong→correct 136; correct→wrong 223. Counts and P-only/C-only transitions appear in `PREDICTION_TRANSITION_AUDIT.csv`.

## Q7: relation to earlier Local-P scope
The earlier Local-P run changed depth1/point1 only. Its subject-equal BA was baseline 0.7738, CE-only 0.7591, and Local-Constrained-P 0.7652; its P-only transplant did not establish a gain. Here both shared blocks can change, but current Shared1 discovery persistence, Shared2 P-only utility, and final native performance all decline. Training objectives also differ, so these results do not isolate parameter count. The previous local arm was not rerun.

## Integrity and verdict
Only the four allowed weight tensors changed; BatchNorm and classifier state remained byte-identical. Geometry hashes match the completed layerwise audit. Outer-dev EEG reads: 0. Final-heldout EEG reads: 0. Verdict: **PERSISTENCE_UTILITY_OBJECTIVE_NOT_SUPPORTED**. This is a seed-0 mechanistic pilot, not a multi-seed performance claim.
