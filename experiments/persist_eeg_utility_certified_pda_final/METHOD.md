# U-PDA method

For frozen feature `z` and population logits `p`, the adapter is `delta(z) = V diag(a) U^T LayerNorm(z) + c`. `a,c` minimize class-balanced label cross-entropy of `p + delta` plus `lambda_A ||a||² + lambda_C ||c||²`; no artificial desired-logit target is used by the primary method. `U,V` are learned from model-fit historical labels only and frozen before validation/outcome fitting.

Each subject's earliest natural session is sorted by archive index and split into four contiguous blocks. Leave-one-block-out adapters produce historical CE utility curves for alpha `{0,.25,.50,.75,1}`. The smallest alpha within one standard error of the best held-block CE is selected. Alpha zero exactly reproduces population. The final adapter uses all allowed history, and future labels/data are metrics-only.
