# LightTail execution and recovery ledger

Existing Windows training PID 25168 was not terminated. New work uses an
isolated LightTail runtime and branch, preserving all original checkpoints.

## Initialization recovery before any candidate training

The current Windows Torch archive hash initially differed from the original
Linux initialization hash. This was not accepted as approximate equivalence.
Inspection identified two container differences: newer `.format_version` and
`.storage_alignment` entries, and numeric vs historical lexicographic storage
record order. The serialization ID additionally contains a platform-dependent
record-name hash. The PyTorch implementation documents the record-name-hash and
CRC components in [writeSerializationId](https://github.com/pytorch/pytorch/blob/v2.5.1/caffe2/serialize/inline_container.cc#L738).

With container differences isolated, ordinary Windows float32 uniform
initialization still did not reproduce the expected hash. Reconstructing the
historical single-rounded float32 uniform transform from the same seeded random
draws did reproduce the entire original initialization SHA256:
`b52f8a94d35a2a2471067a039b658f6f4b351aea9bade4e86615cf463bd472e0`.
All five original folds record this same seed0 hash. No tensor is changed by
the container-hash translation itself: the generated archive is loaded again
and every tensor compared. The uniform override exists only during model
construction and is restored before training. This is exact initialization
recovery, not a new initializer selected by performance.

The Linux source SSH endpoint was unavailable during this turn; all required
selected checkpoints and metadata were already hash-verified on Windows from
the preceding transfer. The original stored episode manifests exist on Windows.

## Preflight and code-level controls

- Corrected an audit-field lookup from `Row.index` to the actual historical
  `Row.cache_index`; no loader or manifest content changed.
- Reconstructed each manifest from the actual historical code and compared full
  JSON identity and LF-normalized SHA against the stored original. Training uses
  the original stored manifest, not the reconstructed copy.
- Initialization, normalizer and all five manifests are gated before formal
  training. Tail0 equivalence tests use real first episodes and both float32 and
  historical AMP, including successful AdamW updates.
- Main loop is the recovered historical function with one loss expression
  replaced, plus observational logging. Optimizer, AMP, clipping, model modes,
  epochs, validation and selected-checkpoint conditions are unchanged.
- Resume only restores matching invariant state; RNG tensors are returned to
  CPU for the CPU RNG API. Trajectory rows beyond the saved training checkpoint
  are excluded on resume, so interrupted output flushing cannot fake updates.

## Interpretation limitation

Current execution uses the available Windows Torch/CUDA environment, not the
original Linux binary environment. Initial tensors and exact sampled episodes
are recovered, and original-CE versus tail0 behavior is tested in this same
current environment. This does not demonstrate bitwise reproduction of all
60 historical CE-training epochs across framework/hardware builds. Report the
result as the requested seed0 diagnostic, not an independent causal proof that
all numerical runtime effects have been eliminated.
