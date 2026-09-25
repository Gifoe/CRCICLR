# Frozen C-direction Actionability and Oracle Audit

## Scope and exclusion audit

This is a development-only audit of four seed-zero EEGNet tasks across folds 0–4. All current experiment EEG reads were explicit `B.rows(..., final=False)` calls for inner-train or discovery subjects. It completed all 20 cells with zero outer-dev EEG reads and zero final-heldout EEG reads. No final-heldout predictions were loaded.

The reused V3 checkpoint, projectors, means, C basis, and utility geometry were frozen before this audit. V3 geometry provenance includes its prior non-final refit pool, which included outer-dev subjects; this audit adds no outer-dev arrays, but its reused geometry is not an independent train-only estimate. This limitation is part of the evidence boundary.

Protocol lock SHA-256: `646d26f15a5544bb748419fc6828001c001daf924865079e21d3981f931f37cd`. Source commit: `0cef590d46d6a004613c66e07555977b3123f8ab`.

## Table 1. Discovery performance and oracle headroom

| Task | Baseline BA | Static V3 BA | Oracle-1dir BA | Oracle-top3 BA | ΔOracle1 | ΔOracle3 |
|---|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | 0.8013 | 0.7978 | 0.8412 | 0.8540 | +3.98 pp | +5.26 pp |
| OpenBMI_ERP | 0.8867 | 0.8861 | 0.9120 | 0.9120 | +2.53 pp | +2.53 pp |
| OpenBMI_SSVEP | 0.9646 | 0.9656 | 0.9743 | 0.9760 | +0.96 pp | +1.14 pp |
| WBCIC_MI | 0.6957 | 0.6973 | 0.7096 | 0.7132 | +1.38 pp | +1.74 pp |

Oracle results use labels only to select oracle actions and are not deployable estimates. Each Δ has a paired 20,000-draw biological-subject bootstrap in `ORACLE_HEADROOM_SUMMARY.csv`. Macro-F1, NLL, rescued errors, and correct predictions damaged are recorded in the decision-level results.

## Table 2. Utility, local response, and trial conditionality

| Task | U+ with negative slope among U+ | P(U+ and negative slope) | P(U+ and α*≤1) | Peak near native | Nonmonotonic | Trial-conditional |
|---|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | 37.7% | 24.3% | 29.9% | 0.0% | 23.8% | 99.8% |
| OpenBMI_ERP | 8.8% | 8.1% | 37.4% | 0.0% | 3.8% | 100.0% |
| OpenBMI_SSVEP | 29.0% | 25.8% | 42.0% | 0.0% | 38.8% | 99.4% |
| WBCIC_MI | 3.9% | 2.2% | 32.2% | 0.0% | 1.2% | 100.0% |

A negative local slope is computed from the fixed centered difference between α=0.75 and α=1.25. Peak-near-native requires the group subject-equal curve maximum in that interval and paired subject-bootstrap lower bounds above both α=0 and α=2. Trial-conditional means the same direction has at least 10% of its trial-level margin optima below α=1 and at least 10% above α=1 within a subject/session.

The four local-actionability quadrants are in `LOCAL_ACTIONABILITY.csv`; the direction-utility relation and margin/CE oracle-alpha distributions are in `UTILITY_VS_ACTIONABILITY.csv` and `TRIAL_CONDITIONALITY.csv`. The detailed trial × direction curves, including margin, CE, prediction, confidence, entropy, and successor-P movement at every locked α, are in each cell’s compressed `RESPONSE_CURVES.csv.gz`; the top-level `RESPONSE_CURVES.csv` is their compact subject/session summary.

## Conditional router

- **OpenBMI_ERP:** router BA 0.8864, baseline 0.8867, static V3 0.8861; triggered oracle `ORACLE_1DIR` BA 0.9120; router gain -0.03 pp; headroom recovered -1.3%.
- **OpenBMI_MI:** router BA 0.8019, baseline 0.8013, static V3 0.7978; triggered oracle `ORACLE_TOP3` BA 0.8540; router gain +0.05 pp; headroom recovered 1.0%.
- **OpenBMI_SSVEP:** router BA 0.9640, baseline 0.9646, static V3 0.9656; triggered oracle `ORACLE_TOP3` BA 0.9760; router gain -0.06 pp; headroom recovered -5.1%.
- **WBCIC_MI:** router BA 0.6958, baseline 0.6957, static V3 0.6973; triggered oracle `ORACLE_TOP3` BA 0.7132; router gain +0.00 pp; headroom recovered 0.2%.

The router fits only inner-train subjects with five-fold subject-grouped OOF diagnostics, then evaluates on disjoint discovery subjects. Inputs are native label-free features. It predicts SUPPRESS/KEEP/ENHANCE for the prelocked top-1 direction and applies α=0.5/1.0/1.5 using the stored frozen response curve. Detailed action metrics and confusion matrices are in `ROUTER_PREDICTION_RESULTS.csv`; applied-model results are in `ROUTER_PERFORMANCE.csv`.

## Interpretation

- `TRIAL_CONDITIONAL_ACTIONABILITY`: oracle BA reaches the 1 pp trigger in at least one task; this establishes development oracle headroom, not deployability.
- `UNPREDICTABLE_ORACLE_HEADROOM`: oracle headroom passed the trigger, but no task router recovered more than 30% on subject-disjoint discovery subjects.

`U_P>0` is an erasure-necessity result. Its relationship to amplification is measured directly by the sign of the local margin slope, the full response curve, and the distribution of trial-specific α optima. Static V3 performance is reported beside both conservative and loose oracle bounds, so a useful direction is not interpreted as a reason to amplify it globally.

## Required artifacts

See `ORACLE_1DIR_RESULTS.csv`, `ORACLE_TOP3_RESULTS.csv`, `ORACLE_UPPER_BOUND.csv`, `ORACLE_HEADROOM_SUMMARY.csv`, `RANDOM_ORACLE_CONTROLS.csv`, `TASK_FOLD_SUMMARY.csv`, `FINAL_HELDOUT_EXCLUSION_AUDIT.json`, and `ROUTER_DATASET_AUDIT.csv`.

Every table is development-only. This report makes no final-heldout performance claim.
