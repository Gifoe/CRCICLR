# Engineering repair ledger

These changes are implementation repairs only.  They do not alter the frozen
Protected assignment, model checkpoints, splits, trial selection, ridge grid,
gating protocol, or any outcome-dependent rule.

| Repair | Scope | Verification | Recovery policy |
|---|---|---|---|
| Broadcast `z0` to trial count before masked TRAIN alpha selection | All cells | Fixed-shape alpha-selection path parses and executes under the same fixed alpha grid. | The initial failed probe JSON was retained under a suffixed archive before its target was rerun. |
| Concatenate gate feature blocks along the feature axis | All cells | Probe reached `COMPLETE`; final transform sanity remained fail-closed. | The first fail-closed probe JSON was retained under a suffixed archive before rerun. |
| Stream NumPy standardization blocks directly into the full device tensor | EEGNet / OpenBMI SSVEP | A deterministic synthetic test matched the historical full standardized dual-ridge prediction **bit-for-bit** (`max_abs=0`).  The final full-matrix GEMMs, ridge alphas, and nested selection remain unchanged. | Pre-repair `MemoryError` JSONs are retained under suffixed archive names; only these engineering failures are rerun, in two-cell GPU batches. |
| Serialize FBCNet / OpenBMI SSVEP execution after a four-worker host failure | FBCNet / OpenBMI SSVEP | The first four-worker attempt produced two Python `MemoryError` closures and two Windows `0xC0000005` native crashes despite ample system memory.  Each serial recovery completed the same frozen cell without retraining or changing an analysis parameter. | The two original JSON failures and two task logs are preserved under `runtime/recovery_archive/fbcnet_openbmi_ssvep_20260921T0849Z`; only the stale failed target JSONs were removed before recovery. |

No final-heldout cohort has been accessed.
