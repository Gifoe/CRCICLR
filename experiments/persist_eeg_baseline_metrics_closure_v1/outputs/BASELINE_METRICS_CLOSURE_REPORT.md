# Baseline metrics closure: protocol-gated stop

Status: **NOT COMPLETE — STOP before new heldout inference, WS-BA aggregation, MAC profiling, or PSWA extension.** This is an audit result, not a manuscript-ready main table. No model was trained, fine-tuned, reselected, or evaluated anew in this closure run.

## Frozen checkpoint audit

The new audit verifies recorded identity, normalizer metadata and the actual SHA-256 of every present `selected.pt`. ModernTCN and Medformer each have 15/15 frozen checkpoints for all four tasks. EEGNet, CBraMod and TeCh also have 15/15 for all four tasks.

LiteBN does **not** have a complete four-task, three-seed matrix in the canonical seven-backbone runtime:

| LiteBN task | Verified checkpoints | Missing |
|---|---:|---|
| OpenBMI MI | 15/15 | — |
| OpenBMI ERP | 15/15 | — |
| OpenBMI SSVEP | 13/15 | fold4 seed1 and seed2 |
| WBCIC MI | 5/15 | seeds 1 and 2 for all five folds |

Under the requested no-training rule, the two incomplete LiteBN cells cannot be presented as complete three-seed primary results. The exact matrix and per-checkpoint paths/hashes are in `BASELINE_COMPLETION_AUDIT.csv` and `FROZEN_CHECKPOINT_MANIFEST_AUDIT.csv`.

## WBCIC cohort conflict — hard regression gate

The prompt's ModernTCN and Medformer WBCIC expected values (BA 0.7177333333333333 and 0.7673) exactly reproduce their existing `HELDOUT_SUBJECT_RESULTS.csv`. However, those results use the ten **V8 internal** development subjects:

`sub-2, sub-3, sub-17, sub-19, sub-21, sub-25, sub-31, sub-33, sub-38, sub-42`.

The established WBCIC **final true outer** subjects are instead:

`sub-4, sub-8, sub-10, sub-15, sub-20, sub-39, sub-40, sub-43, sub-46, sub-51`.

The sets are disjoint. The existing cross-backbone CSGD session artifact uses the true-outer set, and currently contains only seed0 for ModernTCN and Medformer WBCIC, not the prescribed 15-checkpoint matrix. Thus the prompt's numerical regression target is for the wrong biological cohort. A numerical match on the internal cohort cannot validate final true-outer inference. The same problem applies to the corresponding macro-F1 targets. This is documented row by row in `FUTURE_RESULT_REGRESSION_AUDIT.csv`; `protocol/REGRESSION_GATE.json` has `pass=false`.

The OpenBMI expected values match the existing frozen future-session summaries within 1e-8, but that comparison is **not** an independent all-session rerun. The new audit did not access EEG signals or labels. For ModernTCN/Medformer, complete existing all-session results are available only for ModernTCN OpenBMI MI/ERP and Medformer OpenBMI MI; the other model-task matrices require additional inference from frozen checkpoints if the cohort/target conflict is resolved.

## Session notation and PSWA provenance

OpenBMI file S1/S2 map to paper S1/S2. The WBCIC CSGD files are zero-based: file S0/S1/S2 map to paper S1/S2/S3. `protocol/SESSION_MAPPING_AUDIT.json` records both mappings and the two distinct WBCIC cohorts.

The original server checkout at `bd2cc434` is stale: its CSV marks PSWA rows `NOT_ESTIMATED`. The **published GitHub branch** at `3fd6aad5` is newer and contains recovered EEGNet/TeCh values, exact random-control artifacts and recovery code. The published EEGNet/TeCh values match the rounded regression targets in the prompt; their exact values and coverage are copied into `protocol/PSWA_PUBLISHED_REFERENCE.csv` with the source commit. However the same published CSV still marks all ModernTCN and Medformer task rows `INCOMPLETE` with `0/5` coverage. No dirty server worktree output was treated as published provenance, and no ModernTCN/Medformer PSWA was asserted as complete.

## Scientific and execution consequence

No `FINAL_FULLMODEL_METRICS.csv` or manuscript-ready table is issued. In particular, the internal-cohort ModernTCN/Medformer WBCIC BA must not be labelled as final true outer, and LiteBN SSVEP/WBCIC seed0 or 13-checkpoint values must not be labelled three-seed complete. WS-BA, CSGD, bootstrap CIs, paired contrasts, and MACs were not computed after the failed cohort/provenance gate. No original baseline, PEEH, or PSWA output was changed.

To resume this closure without retraining, first provide cohort-consistent frozen true-outer numerical targets for ModernTCN/Medformer (or explicitly amend the regression rule), locate the missing LiteBN frozen checkpoints if they exist, and use the published PSWA recovery implementation and exact random controls to complete ModernTCN/Medformer coverage. If the LiteBN checkpoints do not exist, the agreed six-model four-task three-seed main table is impossible under the no-training constraint; the affected cells must remain explicitly incomplete.
