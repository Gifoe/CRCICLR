# Method

For a training subject `s`, the branch is

`r_s(z) = B[(e_s - mean(e)) * U^T stop_gradient(z)] + (a_s - mean(a))`.

The alternating random-effect step updates only `U,B,e,a` using context
subjects.  The shared step freezes/detaches that branch and updates only the
final feature block and population head using mixed context, population
context, and pseudo-unseen population losses.  `gamma_a=lambda_Q=1`; learning
rates are head `1e-4`, feature block `1e-5`, random effects `1e-3`, weight
decay `1e-3`, and gradient clipping `3.0`.  Inference calls the population
head with a zero random effect and no subject identifier.

