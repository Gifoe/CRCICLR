# Protected-rank audit

Copied from the existing frozen CELL_RESULTS.csv; no decomposition was rerun.

| Task | Full ranks | B1 ranks | Mean Full | Mean B1 | Matched folds |
|---|---|---|---:|---:|---:|
| OpenBMI_MI | [8, 8, 8, 8, 4] | [2, 8, 8, 4, 4] | 7.2 | 5.2 | 3/5 |
| OpenBMI_ERP | [5, 5, 4, 6, 9] | [5, 5, 5, 5, 4] | 5.8 | 4.8 | 2/5 |
| OpenBMI_SSVEP | [4, 4, 4, 8, 8] | [4, 8, 4, 4, 8] | 5.6 | 5.6 | 3/5 |
| WBCIC_MI | [3, 4, 4, 4, 4] | [2, 1, 2, 4, 3] | 3.8 | 2.4 | 1/5 |

WBCIC Full ranks are [3,4,4,4,4] (mean 3.8), whereas B1 ranks are [2,1,2,4,3] (mean 2.4).
B1 does not show higher absolute Protected-only utility under its own selected Protected subspace.
Do not say B1's Protected subspace is intrinsically weaker: Full/B1 selected P are not cross-architecture rank matched.
Within each architecture PSWA is controlled against equal-rank random subspaces; Full PSWA vs B1 PSWA is not cross-architecture rank matched.
