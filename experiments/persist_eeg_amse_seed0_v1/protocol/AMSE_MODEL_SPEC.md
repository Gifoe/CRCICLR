# AMSE-v1 model lock

Exact architecture is recorded in `AMSE_MODEL_SPEC.json`. The anchor H64 path is the only stable input; expressive input is `concat(H15, stop_gradient(H64), H127)`. Final logits are a fixed 50/50 raw-logit average.
