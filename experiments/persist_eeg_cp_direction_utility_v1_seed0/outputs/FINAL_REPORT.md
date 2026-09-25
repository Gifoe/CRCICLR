# Frozen C-direction utility audit

**Scope:** four EEGNet seed-zero tasks, folds 0–4. No backbone, adapter, or gate was trained. TRAIN candidates were frozen before discovery EEG was loaded. `outer_dev_subjects` and every formal final-heldout subject/array were excluded.

**FINAL_HELDOUT_ACCESSED = FALSE**

All direction indices are fold-local. Utilities are functional interventions on a frozen network and do not establish biological causality. `U_int = U_total - U_P - U_C` is reported only as a descriptive nonlinear residual. PCA finite differences are secondary sensitivity results; the primary conclusions use native-coordinate erasure.

## Task and fold summary

| Task | Fold | K(C) | C+ candidates | Validated C+ | C- candidates | Validated C- | Positive utility mass | Negative utility mass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | 0 | 16 | 5 | 1 | 4 | 0 | 0.0715362 | 0.00964335 |
| OpenBMI_MI | 1 | 16 | 7 | 5 | 2 | 0 | 0.0953882 | 0.0134683 |
| OpenBMI_MI | 2 | 16 | 8 | 5 | 3 | 3 | 0.186427 | 0.0200607 |
| OpenBMI_MI | 3 | 16 | 9 | 7 | 1 | 0 | 0.178721 | 0.00724685 |
| OpenBMI_MI | 4 | 16 | 5 | 3 | 3 | 1 | 0.126788 | 0.00949583 |
| OpenBMI_ERP | 0 | 16 | 16 | 15 | 0 | 0 | 0.74899 | 0 |
| OpenBMI_ERP | 1 | 16 | 12 | 10 | 0 | 0 | 0.378415 | 0.0221801 |
| OpenBMI_ERP | 2 | 16 | 15 | 11 | 0 | 0 | 0.617571 | 0.00656305 |
| OpenBMI_ERP | 3 | 16 | 15 | 13 | 1 | 0 | 0.516506 | 0.00236413 |
| OpenBMI_ERP | 4 | 16 | 14 | 13 | 0 | 0 | 0.500548 | 0.000488856 |
| OpenBMI_SSVEP | 0 | 16 | 7 | 7 | 0 | 0 | 0.869885 | 0.0187388 |
| OpenBMI_SSVEP | 1 | 16 | 10 | 7 | 0 | 0 | 1.30248 | 0.0369031 |
| OpenBMI_SSVEP | 2 | 16 | 7 | 3 | 0 | 0 | 0.604224 | 0.024845 |
| OpenBMI_SSVEP | 3 | 16 | 8 | 7 | 2 | 1 | 1.00943 | 0.034441 |
| OpenBMI_SSVEP | 4 | 16 | 12 | 9 | 0 | 0 | 0.933731 | 0.00159068 |
| WBCIC_MI | 0 | 16 | 4 | 3 | 7 | 2 | 0.0101471 | 0.00204881 |
| WBCIC_MI | 1 | 16 | 7 | 6 | 4 | 4 | 0.223713 | 0.0659194 |
| WBCIC_MI | 2 | 16 | 9 | 7 | 4 | 2 | 0.194441 | 0.00977093 |
| WBCIC_MI | 3 | 16 | 11 | 6 | 2 | 0 | 0.0166333 | 0.000266297 |
| WBCIC_MI | 4 | 16 | 15 | 5 | 0 | 0 | 0.0640633 | 0.000464441 |

## Task totals

| Task | Validated C+ count | Validated C- count | C+ P-mediated utility | C- P-mediated utility | Random specificity |
|---|---:|---:|---:|---:|---|
| OpenBMI_MI | 21 | 4 | 0.523553 | -0.0150244 | 5/5 folds exceed random 95th percentile on M+; 3/5 on validated C+ count |
| OpenBMI_ERP | 62 | 0 | 2.53853 | 0 | 5/5 folds exceed random 95th percentile on M+; 3/5 on validated C+ count |
| OpenBMI_SSVEP | 33 | 1 | 4.59326 | -0.0252209 | 5/5 folds exceed random 95th percentile on M+; 4/5 on validated C+ count |
| WBCIC_MI | 27 | 8 | 0.478491 | -0.0580581 | 5/5 folds exceed random 95th percentile on M+; 2/5 on validated C+ count |

## Validated routing breakdown

| Route type | Count | Mean U_P | Mean U_C | Mean U_total |
|---|---:|---:|---:|---:|
| BLOCK_FROM_P_KEEP_C | 5 | -0.00251579 | 0.0114121 | 0.00823835 |
| HARMFUL_OVERALL | 8 | -0.0107156 | -0.0515752 | -0.0606696 |
| ROUTE_TO_P | 38 | 0.0302045 | -0.0103277 | 0.0200763 |
| ROUTE_TO_P_AND_KEEP_C | 105 | 0.0665339 | 0.10442 | 0.208712 |

## Session summaries

`SESSION_DIRECTION_SUMMARY.csv` reports subject-equal direction means separately for each physical session. Candidate bootstraps first average a subject's session means, then resample biological subjects. OpenBMI reports S1/S2; WBCIC reports S0/S1/S2.

## Interpretation

Observed state: **SELECTIVE_CP_ROUTING_SUPPORTED**.

TRAIN identified 229 PCA C+/C- candidates; discovery validated 156. 5 PCA directions were validated as `U_P < 0, U_C > 0`. Random specificity compares each fold's PCA basis with 20 deterministic equal-rank bases in the same spatial complement; the 95th percentile uses the empirical higher quantile across those 20 draws.

A state that says a pattern was supported means the prespecified bootstrap sign repeated in discovery and the PCA summary exceeded the corresponding random-control 95th percentile in at least one fold. This is development evidence for later architecture design, not final generalization evidence.

## Audit artifacts

The per-cell completion manifests hash every analysis file. `PROTOCOL_LOCK.json` fixes roles, checkpoint/projector provenance, estimands, seeds, and thresholds. `DISCOVERY_DIRECTION_LOCK.json` records every TRAIN candidate and basis hash before discovery evaluation. `FINAL_HELDOUT_EXCLUSION_AUDIT.json` records zero heldout EEG reads and zero outer-dev loads.
