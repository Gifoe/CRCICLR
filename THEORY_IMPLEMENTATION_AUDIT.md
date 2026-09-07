# Theory–Implementation Audit

| Requirement | Implementation | Status |
|---|---|---|
| Empirical fixed-future risk target | `critical_index_from_curve`, `critical_index_table` | PASS |
| Full-set sentinel and zero risk | prediction-set and critical-index validators | PASS |
| Alpha-specific predictor | `CriticalIndexPredictor` | PASS |
| Grouped fixed small hyperparameter search | `CriticalIndexPredictor.fit` | PASS |
| Action-only simultaneous residual | `fit_actionwise_simultaneous_quantile` | PASS |
| Higher finite-sample order statistic | `fit_actionwise_simultaneous_quantile` | PASS |
| Ceil and clip certified index | `apply_critical_index_certificate` | PASS |
| U-only post-certificate selection | `select_safe_action` | PASS |
| Sentinel excluded from CSR | selector and simulation fallback tests | PASS |
| Separate alpha guarantees | alpha-specific model and quantile validators | PASS |
| Episode-level, not latent-risk guarantee | `docs/THEORY_SPEC.md` | PASS |
| Legacy Bernstein diagnostic only | diagnostic marker and retired formal predictor API | PASS |
