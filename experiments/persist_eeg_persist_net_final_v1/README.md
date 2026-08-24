# PERSIST-Net final constructive experiment

This directory contains the final theory-derived protection-by-construction
test for PERSIST-EEG.  It starts from commit
`47e773142cc8cd098eeeb89103bebe68c525d760` and never modifies historical
experiments.

Execution is server-only.  Preflight/audit artifacts are created before model
training; runtime checkpoints and the authorized raw cache stay untracked.
Lightweight CSV/JSON results, reports, figures, source-only selection locks,
and per-run certificate audits are committed; raw EEG, runtime caches, and
model checkpoints are not.

## Terminal result

`PERSIST_NET_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED`

Across the frozen 5-fold x 3-seed OpenBMI development matrix, FULL reached
BA 0.76392 and Macro-F1 0.76141. The strongest legal comparable method was
Strong Generic adaptation (BA 0.79242), so the paired subject-level difference
was -2.850 percentage points (10,000-draw bootstrap 95% CI: -4.092 to -1.625
pp). No fold or seed was positive. G1 and G6 failed; WBCIC development and the
sealed outer split were therefore not opened.

Start with `FINAL_REPORT.md`, `results/ablations.csv`, and
`results/statistics.json`. `results/FINAL_ARTIFACT_AUDIT.json` records the final
coverage and purity checks.
