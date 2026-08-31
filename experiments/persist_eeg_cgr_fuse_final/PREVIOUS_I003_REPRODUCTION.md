# PREVIOUS_I003_REPRODUCTION

The historical OpenBMI router cache was re-read and the leave-one-run-out
consensus rule was recomputed before any CGR-Fuse training.  The cache hashes
and recomputed exploration/holdout values are in `results/PREVIOUS_I003_REPRODUCTION.json`.

The independent frozen reference is I003 full-menu ΔBA ≈ +0.008472 and
protected-safe ΔBA ≈ +0.007326 on the 12-subject development holdout.  The
new anchor-free bank uses only complete six-run samples; variable-run samples
remain in the historical audit but are excluded from the new primary metric.
No WBCIC S2 or outer resource was opened.
