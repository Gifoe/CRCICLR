# Frozen current-runtime control protocol

- Model: exact historical `CompactLite` / LiteBN; no architecture, loss,
  optimizer, dropout, BatchNorm, scheduler, checkpoint-rule, manifest or
  normalizer modification.
- Task scope: OpenBMI-MI seed0, canonical folds 0–4, exactly 60 epochs each.
- Initialization: recovered historical bit-level initialization is required
  for every fold and is checked with the project historical serialization hash.
- Training: original stored episode manifests, historical normalizer, AdamW,
  AMP loop and inner-validation selection rule are retained. Outer development
  is not accessed by checkpoint selection.
- Runtime: the existing current Windows/Torch/CUDA configuration is observed
  and recorded; no determinism, TF32, cuDNN or AMP setting is changed to chase
  the historical Linux result.
- Gate: all five folds finish even if early values are low. No model, seed or
  additional dataset is launched afterward. The internal heldout is explicitly
  `INTERNAL_HELDOUT_DIAGNOSTIC_ALREADY_OPEN`, never a sealed or final test.
