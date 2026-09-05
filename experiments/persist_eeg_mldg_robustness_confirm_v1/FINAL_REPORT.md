# Fast MLDG robustness confirmation

Terminal: `EARLY_STOP_MLDG_NOT_ROBUST`

| Dataset | Mean ΔBA | Median ΔBA | Positive cells | Fold0 mean | Fold1 mean |
|---|---:|---:|---:|---:|---:|
| OpenBMI | 1.2381 pp | 0.8571 pp | 3 | 1.2381 pp | nan pp |
| WBCIC | -10.7333 pp | -8.3000 pp | 1 | -10.7333 pp | nan pp |

Stage A passed: `False`.
Compute-matched ERM stop: `False`.

## Direct answers

1. OpenBMI fold0 remains positive across all three optimization seeds (mean +1.2381 pp).
2. WBCIC fold0 does not remain positive (mean -10.7333 pp; only 1/3 cells nonnegative).
3. The earlier WBCIC +16.95 pp was trajectory-dependent; the paired seeds include -24.4 pp and -8.3 pp.
4. Fold consistency cannot be claimed: fold1 was not run after the registered Stage-A stop.
5. Paired median is +0.8571 pp for OpenBMI but -8.3000 pp for WBCIC, so WBCIC fails the +0.25 pp criterion.
6. Compute-matched ERM was not run because Stage A failed; it cannot rescue this result.
7. These data do not support episodic pseudo-unseen optimization as the final method core.
8. Do not enter Final Model design from this candidate; stop this MLDG route unless a new, separately preregistered hypothesis is justified.

The experiment uses frozen Route-B folds, beta values, and refit epochs; no validation selection was rerun.
Canonical outcome labels, OpenBMI sealed holdout, and WBCIC outer-10 were not opened.
The paired-seed result is confirmatory only if all G1--G5 pass on all 12 cells; otherwise it is not sufficient to promote MLDG to a final method.
