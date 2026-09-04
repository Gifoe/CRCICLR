# Bug repair ledger

No scientific-rule repair was made. Engineering repairs in this audit were limited to (1) reloading the frozen tau into a fresh audit context before replay, (2) moving GPU certificate vectors to a common CPU device for deterministic dot products, and (3) wiring calibration/quintile and cluster-bootstrap control gates explicitly. The runner isolates the frozen SSPG helper loader, uses source/refit-only B_out construction, freezes tau before outcomes, and writes explicit invariant flags.
