# RGEO seed0 bug-repair ledger

This ledger records implementation fixes made before the four-task run.

| Item | Symptom | Repair | Scientific effect |
|---|---|---|---|
| Bundle metadata | MI and task-generalisation caches expose row metadata under different field names (`search_rows` vs `rows`). | Added one metadata adapter used only for subject/label lookup. | No data, split, label, or preprocessing change. |
| Geometry BN isolation | A second forward pass could accidentally update BatchNorm statistics. | Geometry forward is wrapped by `model.eval()` and a running-state equality check; gradients remain enabled. | Auxiliary loss receives gradients without a second BN/dropout process. |
| Geometry support | Support selection must not touch validation/outer subjects. | Deterministic support is sampled only from the current fold's inner-train source sessions, with at most two trials per subject/class. | No outer/heldout leakage. |
| Geometry autograd | A detached auxiliary branch would silently make the penalty ineffective. | Unit check requires finite non-zero encoder gradients from `L_geo`. | Confirms the intended loss is trainable. |
| Resume identity | A stale checkpoint could be loaded under a different fold/manifest. | Each latest checkpoint stores the frozen split/manifest and initial-state hashes; mismatch fails closed. | No accidental cross-cell reuse. |
| Heldout ordering | Historical heldout labels must not influence selection. | Development aggregation completes before the one-shot historical heldout loader is called; heldout is not used by the optimizer. | `FINAL_TEST_USED_FOR_TUNING = NO`. |

No scientific rule, fold membership, seed, preprocessing, CE order, optimizer, or geometry formula was changed in response to an outcome.
