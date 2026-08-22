# PERSIST-EEG Experiment 4 — Protection-First Learning

This is the completed EEGNet development audit for the WBCIC/Yang2025/NEMAR `nm000348` primary dataset. The experiment was run on the server GPU and deliberately stopped before the sealed outer set because the prespecified development gate failed.

## Result

`EXP4_PROTECTION_FAILED`

The strong generic adapter had substantial headroom over the frozen S1 anchor (`70.12%` to `77.19%` subject-level S3 BA), but the protected update did not provide a reliable or specific improvement over Generic. The selected V1 hard projection gave `+0.134 pp` Guard−Generic (95% subject bootstrap CI `[-0.085, +0.354] pp`, sign-flip Monte-Carlo `p=0.13663`) and increased negative-transfer rate from `14.6%` to `17.1%`. Random, PCA, persistence-only, and identity controls were not separated from PERSISTGuard.

Three hypothesis-driven variants are retained:

* V1 hard complement projection — simplest preferred formulation and best development mean.
* V2 exact frozen-head decision-response correction — removed decision-response drift, but reduced Guard BA by `0.660 pp` relative to Generic.
* V3 fixed soft response strength `alpha=0.25` — intermediate mechanism drift and `+0.085 pp`, still with CI crossing zero and higher negative transfer.

No outer subject ID, S1/S2/S3 data, or outer membership was opened. No `OUTER_SUBJECT_RESULTS.csv` is produced.

## Layout

`code/` contains the fail-closed executable. `protocol/` contains the development and negative terminal locks. `results/` contains subject-level development tables, controls, mechanism diagnostics, and statistical tests. `iterations/` preserves V2/V3 outputs. `figures/` contains the four development figures.

Raw EEG, epoch caches, and checkpoints are intentionally not included in this repository artifact.
