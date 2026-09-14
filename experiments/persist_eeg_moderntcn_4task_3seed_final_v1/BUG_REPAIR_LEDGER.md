# Bug repair ledger

- 2026-09-15: the first concurrent queue stopped at `OpenBMI_MI/fold3/seed2` with `DefaultCPUAllocator: not enough memory`. The runner retained both the normalized NumPy signal arrays and their identical CUDA tensors for train, validation, and outer-development splits. The signal arrays are now removed from the data dictionary immediately after the unchanged contiguous tensor transfer. Dataset values, ordering, model computation, optimization, checkpoint selection, and metrics are unchanged. Existing cell checkpoints remain resumable.
