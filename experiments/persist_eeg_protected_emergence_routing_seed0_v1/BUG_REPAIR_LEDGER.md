# Engineering repair ledger

These changes are implementation repairs only.  They do not alter the frozen
Protected assignment, model checkpoints, splits, trial selection, ridge grid,
gating protocol, or any outcome-dependent rule.

| Repair | Scope | Verification | Recovery policy |
|---|---|---|---|
| Broadcast `z0` to trial count before masked TRAIN alpha selection | All cells | Fixed-shape alpha-selection path parses and executes under the same fixed alpha grid. | The initial failed probe JSON was retained under a suffixed archive before its target was rerun. |
| Concatenate gate feature blocks along the feature axis | All cells | Probe reached `COMPLETE`; final transform sanity remained fail-closed. | The first fail-closed probe JSON was retained under a suffixed archive before rerun. |
| Stream NumPy standardization blocks directly into the full device tensor | EEGNet / OpenBMI SSVEP | A deterministic synthetic test matched the historical full standardized dual-ridge prediction **bit-for-bit** (`max_abs=0`).  The final full-matrix GEMMs, ridge alphas, and nested selection remain unchanged. | Pre-repair `MemoryError` JSONs are retained under suffixed archive names; only these engineering failures are rerun, in two-cell GPU batches. |

No final-heldout cohort has been accessed.
