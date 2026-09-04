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

## 2026-09-05 — deterministic inner-fold coverage repair

- problem: the first GPU run stopped with `cross-fit held-out coverage failure`.
- cause: `inner_partition` included `k` in the permutation seed, independently
  reshuffling the same source subjects for every slice.
- fix: use one deterministic permutation per dataset/fold/seed and take its
  modulo-5 slices for all held-out sets.
- scientific_definition_changed: false.

## 2026-09-05 — fold-indexed provenance merge repair

- problem: a single-fold worker lock could be written into parent role-hash
  index zero when the parent merged more than one fold.
- fix: map a one-element worker lock to the worker's explicit fold index; no
  training data, weights, outcomes, or model arithmetic are touched.
- scientific_definition_changed: false.

## 2026-09-05 — bounded dynamic worker scheduling

- problem: the paired launcher could leave one GPU slot idle when the two
  dataset workers had unequal runtimes.
- fix: refill a free slot with the other dataset's independent fold while
  keeping at most one OpenBMI and one WBCIC worker resident.  Worker command,
  seed, minibatch order, constants, and serialization are unchanged.
- scientific_definition_changed: false.
