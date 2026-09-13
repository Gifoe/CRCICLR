EXPERIMENT_TYPE = ANALYSIS_ONLY

NEW_MODEL_TRAINED = NO

FINAL_HELDOUT_ACCESSED = NO

INTERNAL_HELDOUT_ACCESSED = NO

DEVELOPMENT_OUTER_ONLY = YES

# Final complementarity report

The oracle values below are label-informed analytical upper bounds, not deployable performance.

| Task | Candidate | Seed | B0 BA | Candidate BA | Net Δ pp | Balanced Rescue pp | Balanced Harm pp | Oracle BA | Oracle Headroom pp | Headroom 95% CI | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| OpenBMI_ERP | LiteBN_X | 0 | 0.818002 | 0.820653 | +0.265 | 5.482 | 5.217 | 0.872822 | 5.482 | [4.588, 6.434] | PROTOCOL_PARTIAL (4/5 folds) |
| OpenBMI_ERP | LiteBN_XS | 0 | 0.832553 | 0.831735 | -0.082 | 5.414 | 5.495 | 0.886689 | 5.414 | [4.557, 6.369] | PASS (5/5 folds) |
| OpenBMI_ERP | LiteBN_XS | 1 | 0.827626 | 0.832992 | +0.537 | 5.311 | 4.774 | 0.880732 | 5.311 | [4.221, 6.389] | PROTOCOL_PARTIAL (3/5 folds) |
| OpenBMI_ERP | LiteBN_XS | 2 | 0.824823 | 0.824154 | -0.067 | 4.806 | 4.872 | 0.872879 | 4.806 | [3.884, 5.765] | PROTOCOL_PARTIAL (3/5 folds) |
| OpenBMI_MI | LiteBN_X | 0 | 0.784750 | 0.793000 | +0.825 | 8.975 | 8.150 | 0.874500 | 8.975 | [7.425, 10.575] | PASS (5/5 folds) |
| OpenBMI_MI | LiteBN_XS | 0 | 0.791500 | 0.799000 | +0.750 | 9.275 | 8.525 | 0.884250 | 9.275 | [7.850, 10.775] | PASS (5/5 folds) |
| OpenBMI_MI | LiteBN_XS | 1 | 0.783250 | 0.750750 | -3.250 | 8.400 | 11.650 | 0.867250 | 8.400 | [6.750, 10.076] | PASS (5/5 folds) |
| OpenBMI_MI | LiteBN_XS | 2 | 0.783750 | 0.778750 | -0.500 | 8.725 | 9.225 | 0.871000 | 8.725 | [7.525, 10.000] | PASS (5/5 folds) |
| OpenBMI_SSVEP | LiteBN_X | 0 | 0.892000 | 0.913250 | +2.125 | 5.525 | 3.400 | 0.947250 | 5.525 | [3.150, 8.300] | PASS (5/5 folds) |
| OpenBMI_SSVEP | LiteBN_XS | 0 | 0.892000 | 0.933250 | +4.125 | 6.650 | 2.525 | 0.958500 | 6.650 | [3.600, 10.600] | PASS (5/5 folds) |
| OpenBMI_SSVEP | LiteBN_XS | 1 | 0.904750 | 0.909250 | +0.450 | 4.525 | 4.075 | 0.950000 | 4.525 | [2.375, 7.200] | PASS (5/5 folds) |
| OpenBMI_SSVEP | LiteBN_XS | 2 | 0.925250 | 0.909000 | -1.625 | 3.225 | 4.850 | 0.957500 | 3.225 | [1.725, 5.025] | PASS (5/5 folds) |
| WBCIC_MI | LiteBN_X | 0 | 0.790274 | 0.783670 | -0.660 | 7.583 | 8.243 | 0.866101 | 7.583 | [5.806, 9.391] | PASS (5/5 folds) |
| WBCIC_MI | LiteBN_XS | 0 | 0.787040 | 0.789935 | +0.289 | 8.438 | 8.148 | 0.871418 | 8.438 | [6.698, 10.274] | PASS (5/5 folds) |
| WBCIC_MI | LiteBN_XS | 1 | 0.784142 | 0.790721 | +0.658 | 7.775 | 7.117 | 0.861892 | 7.775 | [6.115, 9.420] | PASS (5/5 folds) |
| WBCIC_MI | LiteBN_XS | 2 | 0.775118 | 0.795729 | +2.061 | 8.954 | 6.892 | 0.864654 | 8.954 | [6.710, 11.389] | PASS (5/5 folds) |

## Descriptive paired rates

| Task | Candidate | Seed | Rescue/all | Harm/all | Rescue given B0 wrong | Harm given B0 correct |
|---|---|---:|---:|---:|---:|---:|
| OpenBMI_ERP | LiteBN_X | 0 | 0.059170 | 0.041051 | 0.403857 | 0.048098 |
| OpenBMI_ERP | LiteBN_XS | 0 | 0.057146 | 0.046086 | 0.417566 | 0.053393 |
| OpenBMI_ERP | LiteBN_XS | 1 | 0.052483 | 0.043792 | 0.388474 | 0.050633 |
| OpenBMI_ERP | LiteBN_XS | 2 | 0.047769 | 0.040551 | 0.373232 | 0.046503 |
| OpenBMI_MI | LiteBN_X | 0 | 0.089750 | 0.081500 | 0.416957 | 0.103855 |
| OpenBMI_MI | LiteBN_XS | 0 | 0.092750 | 0.085250 | 0.444844 | 0.107707 |
| OpenBMI_MI | LiteBN_XS | 1 | 0.084000 | 0.116500 | 0.387543 | 0.148739 |
| OpenBMI_MI | LiteBN_XS | 2 | 0.087250 | 0.092250 | 0.403468 | 0.117703 |
| OpenBMI_SSVEP | LiteBN_X | 0 | 0.055250 | 0.034000 | 0.511574 | 0.038117 |
| OpenBMI_SSVEP | LiteBN_XS | 0 | 0.066500 | 0.025250 | 0.615741 | 0.028307 |
| OpenBMI_SSVEP | LiteBN_XS | 1 | 0.045250 | 0.040750 | 0.475066 | 0.045040 |
| OpenBMI_SSVEP | LiteBN_XS | 2 | 0.032250 | 0.048500 | 0.431438 | 0.052418 |
| WBCIC_MI | LiteBN_X | 0 | 0.075868 | 0.082486 | 0.361538 | 0.104392 |
| WBCIC_MI | LiteBN_XS | 0 | 0.084423 | 0.081517 | 0.396212 | 0.103590 |
| WBCIC_MI | LiteBN_XS | 1 | 0.077805 | 0.071186 | 0.360239 | 0.090797 |
| WBCIC_MI | LiteBN_XS | 2 | 0.089588 | 0.068927 | 0.398135 | 0.088940 |

## XS cross-seed summary

| Task | Mean net Δ pp | Mean oracle headroom pp | Min | Max | Mean balanced rescue pp | Mean balanced harm pp | All seeds >=1 pp | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| OpenBMI_ERP | +0.129 | 5.177 | 4.806 | 5.414 | 5.177 | 5.047 | NO | PROTOCOL_PARTIAL |
| OpenBMI_MI | -1.000 | 8.800 | 8.400 | 9.275 | 8.800 | 9.800 | YES | PASS |
| OpenBMI_SSVEP | +0.983 | 4.800 | 3.225 | 6.650 | 4.800 | 3.817 | YES | PASS |
| WBCIC_MI | +1.003 | 8.389 | 7.775 | 8.954 | 8.389 | 7.386 | YES | PASS |

## Seed0 B0+X+XS union oracle

The common B0 is the exact X seed-0 historical B0 cell. The XS candidate checkpoint and normalizer are identical to the seed-0 XS development replay; this avoids mixing incompatible baselines.

| Task | B0 BA | X oracle headroom pp | XS oracle headroom on common B0 pp | Union headroom pp | Gain over best single pp | X-only rescue | XS-only rescue | Both rescue | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenBMI_ERP | 0.818002 | 5.482 | 5.685 | 7.557 | 1.872 | 1061 | 1199 | 2688 | PROTOCOL_PARTIAL (4/5 folds) |
| OpenBMI_MI | 0.784750 | 8.975 | 9.575 | 11.850 | 2.275 | 91 | 115 | 268 | PASS (5/5 folds) |
| OpenBMI_SSVEP | 0.892000 | 5.525 | 6.650 | 7.250 | 0.600 | 24 | 69 | 197 | PASS (5/5 folds) |
| WBCIC_MI | 0.790274 | 7.583 | 7.904 | 10.131 | 2.227 | 138 | 158 | 332 | PASS (5/5 folds) |

## Replay exclusions

5/80 cells failed the fixed deterministic full-FP32 replay gate and were excluded from every statistic. Partial rows are descriptive estimates over the remaining eligible outer subjects, not complete task estimates.

| Comparison | Task | Seed | Fold | Status |
|---|---|---:|---:|---|
| X | OpenBMI_ERP | 0 | 1 | PROTOCOL_FAIL_FOR_CELL |
| XS | OpenBMI_ERP | 1 | 0 | PROTOCOL_FAIL_FOR_CELL |
| XS | OpenBMI_ERP | 1 | 3 | PROTOCOL_FAIL_FOR_CELL |
| XS | OpenBMI_ERP | 2 | 0 | PROTOCOL_FAIL_FOR_CELL |
| XS | OpenBMI_ERP | 2 | 3 | PROTOCOL_FAIL_FOR_CELL |

## Scientific interpretation

1. X has >=1.0 pp oracle rescue headroom on 3/4 tasks; meaningful complementarity is present on the development outer cohort.
2. XS has >=1.0 pp oracle rescue headroom on 10/12 task-seed cells; meaningful complementarity is present.
3. XS clears 1.0 pp in every seed on 3/4 tasks; other task signals are seed-isolated or below the practical threshold.
4. Per-task/seed oracle headroom is reported in the primary table; cross-seed ranges are reported in the XS table.
5. The seed0 union exceeds the better single-candidate oracle on 3/3 complete tasks; the partial ERP estimate is reported separately and is not counted as complete evidence.
6. Based only on oracle headroom, the next stable-rescue/predictability analysis is worth investigating for 6 candidate/task cases under the stated 1.0 pp threshold; this audit launches no next-stage model.
