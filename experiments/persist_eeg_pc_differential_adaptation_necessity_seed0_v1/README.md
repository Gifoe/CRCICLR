# Frozen P/C differential adaptation necessity (seed 0)

This experiment tests whether two adaptation rates for source-derived `PROTECTED` and its orthogonal complement transfer better than the TRAIN-selected best homogeneous rate. It does not test a new backbone or an absolute freeze-P rule. `PERSIST_ALL` and three rank-matched alternative families are controls. All adaptation is unlabeled diagonal affine recalibration at the frozen final-classifier input; model weights, native head and normalization buffers stay frozen.

The locked population split and selection rules are in `protocol/PROTOCOL_LOCK.json`. Fold artifacts are stored outside Git on the original server. Formal final-heldout EEG is never opened. Only frozen `OUTER_DEVELOPMENT` subjects can be used for held evaluation, never for constructing a basis or tuning parameters. The historical protection-first negative result and the shared-decision-geometry negative result are preserved as constraints, not overwritten.

This branch includes exactly EEGNet/OpenBMI_MI, EEGNet/OpenBMI_ERP and EEGConformer/OpenBMI_MI, seed 0, folds 0–4. A missing or zero-rank Protected basis is recorded, never fabricated. The terminal report presents negative results without substituting a target-label oracle for a deployable method.

## Completed result

All 15 folds completed on the original server. The primary EEGNet/OpenBMI_MI TRAIN selector was off-diagonal in only 3/5 folds. Its OUTER_DEVELOPMENT differential-minus-best-uniform BA was +0.00075 with a paired biological-subject 95% bootstrap CI of [-0.00175, +0.00325]. EEGNet/ERP and EEGConformer/MI did not replicate a consistent held advantage. The locked decision is `UNIFORM_SHRINKAGE_EXPLAINS_THE_GAIN`; next action `STOP_PC_SPECIFIC_ADAPTATION`. This is not evidence that P is invariant or C is nuisance. Formal final-heldout EEG reads remained zero.

The complete numerical answer, including all 15 fold rows, is in `outputs/FINAL_REPORT.md`; compact machine-readable results are in `outputs/`. `protocol/SOURCE_PROVENANCE.json`, `REPRESENTATION_PROVENANCE.json`, and `PC_GEOMETRY_AUDIT.json` carry the checkpoint/source/split/normalizer/frozen-state and source-only basis audits. `code/validate_compact.py` independently checks the delivered inventories, row counts, identities, cross-file arithmetic and heldout exclusion.

## Reproduction boundary

Server execution was deliberately staged: `code/run.py` prepared TRAIN embeddings and source-only P/C geometry; `code/analysis.py` selected the operator, context budget, and partition-specific rates using TRAIN pseudo-targets; `code/outer.py` extracted OUTER_DEVELOPMENT only after all 15 TRAIN selections were frozen, then evaluated fixed configurations; `code/aggregate.py` produced the compact decision. The numbered PowerShell wrappers are the original-server one-heavy-lane launch guards; the executed OUTER wrapper was `run_outer_queue_v3.ps1`. Earlier uploaded OUTER v1/v2 wrappers were never launched and produced no scientific outputs. The final OUTER code takes ERP context by frozen 12-stimulus cache order without reading its labels for adaptation. The formal final-heldout population was never accessed.

Large TRAIN embedding caches, raw per-fold OUTER JSON, checkpoints, and execution logs remain on the original server and are intentionally excluded from Git. The checked-in compact outputs are byte-identical to those generated there.
