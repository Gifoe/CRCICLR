# Bug repair ledger

Training settings and data scope were not changed after outcome access. Two packaging-only defects were repaired in the final artifact layer: the optional `tabulate` dependency was absent in MNElab, and the report writer selected a grouped fold table instead of the already-written pivot table. `finalize_sspg_seed0.py` reads only compact outputs and corrects validation/report serialization; the pre-outcome training lock and runner code hash remain unchanged.
