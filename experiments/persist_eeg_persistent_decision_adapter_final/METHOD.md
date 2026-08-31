# Method

For subject `s` and historical block `k`, the adapter is
`V diag(a_s + a_s,k^transient) U^T LayerNorm(z) + c_s + c_s,k^transient`.
`U,V` are shared low-rank bases estimated from historical model-fit blocks.
Each block code is fitted by a closed-form ridge correction against the frozen
population logits. Persistent codes use the declared diagonal-Fisher formula
`(lambda_precision I + sum F)^-1 sum(F a)`, with the analogous intercept
pool. Leave-one-block-out estimates define the cross-fit diagnostic. Transient
residuals are explicitly centered. Subject-balanced metrics and paired
biological-subject bootstrap are used throughout. No future label or gradient
enters fitting, pooling, or recipe selection.
