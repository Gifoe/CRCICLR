# Bug-fix log

- Before audit completion, the BIDS-sidecar fetcher was made retryable and restart-cacheable after a transport timeout. It still reads the same version-pinned official sidecars, does not alter labels, cache content, epoch definitions, splits, models, or outcomes.

- Seed-0 runner initially looked for a top-level `seeds` field although the already-committed amendment stores it under `scientific_changes`. It failed before loading data or training; the schema lookup was corrected and the amendment hash was refreshed. No protocol value or scientific outcome changed.

- Resuming a CUDA-mapped checkpoint failed before optimization because PyTorch requires the saved CPU RNG state on CPU. `restore_rng` now applies `.cpu()` to that state only. This exactly restores the serialized state; no model/data/split/loss/optimizer/seed/metric definition changed.

- CUDA RNG state has the same PyTorch CPU-ByteTensor API requirement when restored from a CUDA-mapped checkpoint. The resume path now converts saved CUDA RNG tensors to CPU ByteTensors before `set_rng_state_all`; no numerical training definition changed.

- Final aggregation initially resolved the fixed historical MI reference table one directory too high. The path now points to `experiments/persist_eeg_final_heldout_confirmation_v1/outputs/DATASET_RESULTS.csv`; this changes no task data, trained model, metric, or terminal rule.

- Final Markdown aggregation attempted to cast the display-string positive-subject field (for example `14/14`) to an integer. The formatter now renders the already-computed display value directly; no data, statistic, or terminal criterion changed.
