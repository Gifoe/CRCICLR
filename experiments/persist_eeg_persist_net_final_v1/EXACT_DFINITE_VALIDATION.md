# Exact D_finite validation

The implementation copies the archived Exp3/DDA definition:

`sqrt(mean(sum((delta_logits - mean_class(delta_logits))^2, class)))`.

For a binary margin displacement `d`, the analytic equivalent is
`sqrt(mean(d^2)/2)`.  The preflight test requires <=1e-12 absolute error and
also verifies the frozen archived Exp3 reference values.  Numeric results are
written before training to `protocol/EXACT_DFINITE_VALIDATION.json`.
