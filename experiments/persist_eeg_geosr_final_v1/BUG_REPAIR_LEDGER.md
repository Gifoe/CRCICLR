# Bug repair ledger

## 2026-09-05 — explicit CUDA index restores resident-cache reuse

- cause: `torch.device('cuda') != torch.device('cuda:0')`. `FoldCache.tensor`
  therefore re-uploaded the full source tensor and recomputed its normalized
  view on every batch when invoked with the unindexed device.
- fix: launch with the explicit current CUDA index. Keep `run_geosr.py` and
  `audit_primitives.py` byte-identical so previously validated cache hashes
  remain reusable. The full-protocol accelerated launcher uses the same fix.
- source-only full-epoch verification, both datasets fold0: identical loss
  and identical full model-state SHA-256 before and after the device change.
  OpenBMI 25.4026 -> 1.4519 sec/epoch (17.50x); WBCIC 70.2997 -> 1.9049
  sec/epoch (36.91x). See `DEVICE_CACHE_EQUIVALENCE.json`.
- change RAPID_TRIAGE scheduling to one worker on the single GPU. Its previous
  concurrent epoch times were approximately 51 s and 149 s, respectively.
- preserve atomic progress files, honor already-satisfied early stopping on
  resume, return actual checkpoint hashes on cache hits, verify hashes before
  outcome access, and report elapsed epochs rather than selected-best epochs.
- numerical resume/early-stop regressions and existing protocol tests: 7 passed.
- scientific_definition_changed: false for these engineering fixes. The
  separately locked RAPID_TRIAGE amendment changes scientific scope and remains
  a directional screen, not a formal seed0 result.

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
