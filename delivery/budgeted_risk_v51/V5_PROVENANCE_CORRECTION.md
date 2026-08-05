# V5 provenance correction

The old stop narrative stating that Stage-0A failed and budget experiments were not opened is incorrect. The authoritative `STAGE0_DECISION.json` records `full_context_pass=true`, `budget_experiments_opened=true`, and `budget_pass=false`. Stage-0A therefore passed; Stage-0B ran and failed its efficiency gate.

This is a reporting/provenance error only. It does not alter the stored numerical results or the preregistered `STAGE0_NO_GO` verdict. Source-of-truth priority is: STAGE0_DECISION.json, RUN_STATE.json, BUDGET_GATE_SUMMARY.csv, FULL_CONTEXT_UPPER_BOUND.md, RANDOM_BUDGET_BASELINE.md, then any stop narrative. The old files are retained unchanged.
