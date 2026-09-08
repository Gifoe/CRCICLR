# Bug-fix log

- Before audit completion, the BIDS-sidecar fetcher was made retryable and restart-cacheable after a transport timeout. It still reads the same version-pinned official sidecars, does not alter labels, cache content, epoch definitions, splits, models, or outcomes.

- Seed-0 runner initially looked for a top-level `seeds` field although the already-committed amendment stores it under `scientific_changes`. It failed before loading data or training; the schema lookup was corrected and the amendment hash was refreshed. No protocol value or scientific outcome changed.

- Resuming a CUDA-mapped checkpoint failed before optimization because PyTorch requires the saved CPU RNG state on CPU. `restore_rng` now applies `.cpu()` to that state only. This exactly restores the serialized state; no model/data/split/loss/optimizer/seed/metric definition changed.

- CUDA RNG state has the same PyTorch CPU-ByteTensor API requirement when restored from a CUDA-mapped checkpoint. The resume path now converts saved CUDA RNG tensors to CPU ByteTensors before `set_rng_state_all`; no numerical training definition changed.
