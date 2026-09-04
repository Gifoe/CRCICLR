# Bug repair ledger

No scientific repair was performed.  The implementation was written as a
separate experiment from the audit base.  Any subsequent engineering-only
repair (path, device, memory, serialization, or deterministic execution) must
be appended here with `scientific_definition_changed = false` before rerun.

## 2026-09-05 — GPU data-view optimization

- problem: the first smoke launch spent most time transferring normalized EEG
  batches from CPU to GPU (observed GPU utilization about 2%).
- cause: the source tensor view was indexed on CPU for every minibatch.
- fix: retain the source-only mmap materialization in RAM and construct one
  active normalized CUDA view per teacher/student normalizer; minibatches are
  then indexed directly on the GPU.  The view is discarded/replaced whenever
  the frozen A_k normalizer changes.
- scientific_definition_changed: false.
