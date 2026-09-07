# Result ledger

Server closure (2026-08-22):

* Code commits: `77a650ab` (train-only matching repair), `8b77b0be` (identity
  curve column bug fix), `1f35aea5` (honest G1 terminal path).
* G0 held-out persistence: pass; mean `R_persist=0.5662742334`, bootstrap
  95% CI `[0.4345339074, 0.7036389605]`, 23 unique validation subjects.
* G1: fail for fold 2 / seed 1.  Protected rank is 8, the frozen
  non-Protected persistence-supported pool has size 8, and there is only one
  legal exact-rank candidate; the required minimum is 20.
* G2 and G3: not evaluated.  No validation task outcome was used to alter the
  design, and no `H_P`, `H_N` or `Delta_H` value is reported.
* Terminal state: `MATCHED_NONPROTECTED_CONTROL_UNAVAILABLE`.
* Utility-not-identity claim: `PARTIAL`; Experiment-4 entry: `NO`.
* Outer data and membership: untouched (`false` for both flags).

The terminal phase was run twice; the lightweight output hashes in
`outputs/REPRODUCIBILITY_AUDIT.json` were identical.
