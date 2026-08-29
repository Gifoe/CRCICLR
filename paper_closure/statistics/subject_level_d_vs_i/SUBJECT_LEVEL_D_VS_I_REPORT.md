# Subject-level Decision Dependence versus Identity reanalysis

Protocol: `PERSIST_EEG_SUBJECT_LEVEL_D_VS_I_REANALYSIS_V1`  
Pre-outcome lock commit: `d40d64812982044324bab2030b40ed5cc5100d3b`  
Statistical unit: **biological subject**

This report is the binding post-hoc subject-level validity repair. Seeds, folds, configurations,
directions, trials, runs, and backbone rows are repeated measurements, not independent subjects.

## Results

| Dataset | Primary analysis | Subjects | mean RMSE I | mean RMSE D | mean subject Δ(I−D) | refitted subject-bootstrap 95% CI | D better (subjects) | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| OPENBMI_STRESS | `OPENBMI_EQUAL_FAMILY_PRIMARY` | 40 | 0.04919640 | 0.04832046 | 0.00087594 | [-0.00135110, 0.00253199] | 23/40 | `PARTIAL` |
| WBCIC_REPLICATION | `WBCIC_ERM_PRIMARY` | 41 | 0.02583074 | 0.02486550 | 0.00096525 | [-0.00227637, 0.00450375] | 21/41 | `PARTIAL` |

Cross-dataset terminal: **`CROSS_DATASET_PARTIAL`**.

Δ(I−D) is computed within each subject as RMSE(Identity model) minus RMSE(Decision model);
positive values favor Decision Dependence. OpenBMI backbone-specific deltas are averaged
within subject before any inference.

The OpenBMI primary is the predeclared equal-family full-grid estimand; the WBCIC primary
is the frozen ERM bank. They are two dataset-specific estimands, not an exactly matched
cross-dataset intervention-bank replication. The OpenBMI ERM-only result is reported below
as the mandatory direct-scope sensitivity.

### Backbone wording gate

OpenBMI primary backbone point estimates are both positive: **FALSE**.
The phrase 'across two OpenBMI backbones' is permitted only when this value is TRUE.

### Mandatory analyses and sensitivities

| Analysis | Role | Subjects | mean subject Δ(I−D) | 95% CI | Gate |
|---|---|---:|---:|---:|---|
| `OPENBMI_EQUAL_FAMILY_PRIMARY` | `PRIMARY_OPENBMI` | 40 | 0.00087594 | [-0.00135110, 0.00253199] | `PARTIAL` |
| `OPENBMI_EQUAL_CONFIG_SECONDARY` | `MANDATORY_GRID_WEIGHTING_SENSITIVITY` | 40 | 0.00114502 | NOT COMPUTED (point-only sensitivity) | `POINT_ONLY_MANDATORY_SENSITIVITY` |
| `OPENBMI_ERM_ONLY_SENSITIVITY` | `MANDATORY_DIRECT_WBCIC_SCOPE_SENSITIVITY` | 40 | -0.00210273 | NOT COMPUTED (point-only sensitivity) | `POINT_ONLY_MANDATORY_SENSITIVITY` |
| `WBCIC_ERM_PRIMARY` | `PRIMARY_WBCIC` | 41 | 0.00096525 | [-0.00227637, 0.00450375] | `PARTIAL` |

## Scope and interpretation

- This uses already-observed development outcomes only and is not an independent confirmation.
- OpenBMI internal-14, the OpenBMI policy holdout, and WBCIC outer-10 were not accessed.
- Exp3 remains exact but run-level/algorithmic because its per-subject runtime was not retained.
- No sign-flip p-value was computed because exchangeability is not established for the overlapping
  peer-subject fits.
- `POINT_CI_DIRECTION_CONFLICT` is a predeclared conservative terminal for a point estimate and
  percentile interval lying entirely in opposite directions; it permits no directional claim.
- The narrow estimand is whether source-side Decision Dependence better predicts held-subject
  intervention consequence for held algorithmic runs within the frozen intervention bank.
- It does not establish a causal mechanism, unseen-intervention transfer, deployment utility,
  learning-algorithm population performance, or that Decision Dependence guarantees utility.
- The predeclared outcome sign is binding; no alternative feature, fold, alpha, or aggregation
  was searched after reconstruction.
