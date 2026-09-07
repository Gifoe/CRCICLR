# Per-seed early-stop MLDG check

Terminal: `MLDG_PER_SEED_EARLY_STOP_PARTIAL_ONLY`

| Seed | ERM epoch | MLDG epoch | ERM BA | MLDG BA | ΔBA |
|---:|---:|---:|---:|---:|---:|
| 0 | 9 | 20 | 74.6000 | 55.2000 | -19.4000 |
| 1 | 7 | 17 | 50.0000 | 75.5000 | +25.5000 |
| 2 | 11 | 12 | 50.5000 | 66.3000 | +15.8000 |

Mean ΔBA: `+7.3000 pp`; median: `+15.8000 pp`; positive seeds: `2/3`; minimum: `-19.4000 pp`; any < -5 pp: `True`.

## Answers

1. Selected epochs differ across seeds: see PER_SEED_SELECTION.csv.
2. Epoch-6 versus post-epoch-6 source-validation BA is in MLDG_EPOCH6_DIAGNOSTIC.csv; held-out BA was not used for selection.
3. Per-seed legal early stopping terminal is `MLDG_PER_SEED_EARLY_STOP_PARTIAL_ONLY`.
4. This single WBCIC fold cannot support a broader final-model claim.
5. If the failure terminal is present, stop plain MLDG and require a new preregistered hypothesis.

Only WBCIC fold0, EEGNet, ERM/MLDG, and optimization seeds 0/1/2 were run. Split/cache were reused from Route-B; no canonical outcome or WBCIC outer-10 was opened.
