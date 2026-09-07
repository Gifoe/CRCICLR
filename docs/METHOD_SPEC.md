# HSC-TTA method specification

The formal method is defined in `THEORY_SPEC.md`. Its target is empirical miscoverage on the immutable `V_s-main` episode. It uses an alpha-specific critical-index predictor and a subject-level conformal residual maximized over three actions only.

Prediction sets are `C_lambda(x)={k:p_k>=1-lambda} union {argmax_k p_k}`. The grid has 20 nontrivial values from 0.50 through 0.99 and a `lambda=1.0` full-set sentinel. Curves must be nested, sentinel risk must be zero, and the sentinel is excluded from nontrivial CSR.

Action selection reads certified indices and context-only utility diagnostics. It rejects tables containing future outcome fields and requires `n_classes`; there is no infinity default. The deterministic order is minimum context average set size, maximum context singleton rate, minimum fixed action cost, more conservative certified lambda, then action name.

The former upper-risk regression and risk-space action×lambda maximum are retired from the formal pipeline. `empirical_bernstein_bound` is retained only as a legacy supplementary diagnostic and carries an explicit diagnostic marker.
