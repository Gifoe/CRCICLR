# Signed Audit V3.1

Reproducibility repair for Signed Audit V3. The scientific utility, thresholds,
five subject-disjoint inner splits, 100 random draws, and 10,000 bootstrap
draws are unchanged. V3.1 replaces process-dependent sampling with SHA256
seeds, persists every selected index, and stores one canonical full-TRAIN
persistence spectrum per fold/seed.

`results_v3_1/SIGNED_V3_1_FINAL_REPORT.json` reports
`PERSISTENCE_UTILITY_ASSIGNMENT_REPRODUCIBLE`: all 6/6 MI runs have Protected
assignments, all 6/6 exceed matched-rank random harm, and the hierarchical
95% CI for the harm difference is `[0.0679181, 0.1230136]` BA.

The V3.1 self-test passed exact replay for MI, ERP, SSVEP validation effects
and five block-level utilities. These artifacts are the only basis permitted
for Shared Geometry V1.2.
