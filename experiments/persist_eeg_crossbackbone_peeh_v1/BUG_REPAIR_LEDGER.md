# Bug repair ledger

- Preflight found SGN had only 14/20 frozen seed-0 checkpoint cells. SGN was
  deferred rather than trained or silently substituted.
- The first EEGNet smoke run exposed NumPy-array truth testing in the erasure
  helper. The guard was changed from `if not dims` to `if len(dims) == 0`.
  This is an execution repair and does not alter any numeric operation.
- TFFormer is absent by explicit user instruction, not because of a runtime
  failure.
