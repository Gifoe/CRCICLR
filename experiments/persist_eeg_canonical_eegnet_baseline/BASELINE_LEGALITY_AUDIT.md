# Baseline legality audit

## CANONICAL BASELINE STATUS

PASS: all frozen role, session and row-count assertions passed. OpenBMI uses 54 Stage-0 subjects. WBCIC uses only the 41 subjects in `DEVELOPMENT_SCOPE_LOCK.json`; its outer cohort is not enumerated. Initial fits use model-fit S1+S2, discovery is disjoint, refits use model-fit+discovery S1+S2, and outcome scoring occurs only after refit.

The runner constructs outcome indices only for the final scoring call and never uses outcome labels for training, normalization, or epoch selection.
