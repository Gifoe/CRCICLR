# V4 final development research model specification

Terminal state: `GENERIC_DYNAMIC_ENSEMBLE_WINS`

The selected discovery model is a positive, availability-normalized pool over
six frozen KEEP margins. It learns one global scale and bias. L2 is selected
from `{0.01, 0.1, 1.0}`; the inner threshold is restricted to
`{0.475, 0.500, 0.525}`. All selection is subject-disjoint.

## Mean OpenBMI fold weights

- `fold-0_seed-0`: `0.221141`
- `fold-0_seed-1`: `0.202222`
- `fold-1_seed-0`: `0.197788`
- `fold-1_seed-1`: `0.116823`
- `fold-2_seed-0`: `0.110291`
- `fold-2_seed-1`: `0.151735`

ACTION experts and PERSIST inputs are excluded from the selected performance
model because their matched incremental effects are non-positive. This is a
development-research freeze, not an outer-evaluation authorization. The WBCIC
direct transfer was negative and the best adapted generic linear stack had a
confidence interval crossing zero.

- Outer subject IDs loaded: `false`
- `OUTER_TEST_USED=false`
