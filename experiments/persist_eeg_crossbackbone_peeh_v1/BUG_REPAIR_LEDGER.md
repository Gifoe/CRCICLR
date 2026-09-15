# Bug repair ledger

- Preflight found SGN had only 14/20 frozen seed-0 checkpoint cells. SGN was
  deferred rather than trained or silently substituted.
- The first EEGNet smoke run exposed NumPy-array truth testing in the erasure
  helper. The guard was changed from `if not dims` to `if len(dims) == 0`.
  This is an execution repair and does not alter any numeric operation.
- TFFormer is absent by explicit user instruction, not because of a runtime
  failure.
- Protocol audit found that the initial ridge helper fitted a new feature mean
  and standard deviation after each erasure.  This contradicted the locked
  requirement to fit the standardizer on the intact training representation
  and hold it fixed across interventions.  The initial 59 runtime cells were
  quarantined outside git as
  `crossbackbone_peeh_runtime_protocol_invalid_20260915`; they are not reused.
- The same helper rebuilt a full original-D Gram matrix for every intervention.
  This is pathological for ModernTCN's 230,144-dimensional head input.  The
  repaired runner fits the intact standardizer once per probe split, computes
  original-D products once, and expresses each erasure as a rank-at-most-2d
  kernel update solved with Woodbury.  Compact D<=N representations retain a
  direct primal solve.  Random-erasure count, split count, cap, active rank,
  alpha, seeds, checkpoint identity, model mode, and data are unchanged.
- `test_fast_ridge.py` checks the low-rank dual path against an explicit kernel
  construction and checks both the dual and compact primal paths against a
  slow raw-erasure reference with the same fixed intact standardizer.
