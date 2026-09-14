# Bug repair ledger

- 2026-09-15: applied the numerically equivalent concurrent-memory repair identified in the matched ModernTCN queue. After the unchanged contiguous device transfer, redundant NumPy signal arrays are released instead of being retained alongside CUDA tensors. Dataset values, ordering, model computation, optimization, checkpoint selection, and metrics are unchanged.
- 2026-09-15: made checkpoint RNG restoration device-safe by moving saved ByteTensor states back to CPU before passing them to PyTorch's RNG-state APIs. This preserves the exact saved random state and affects only interrupted-cell resume.
- 2026-09-15: resume checkpoints are now deserialized on CPU; only live model and optimizer tensors are transferred by their standard load operations. This avoids redundant CUDA copies in concurrent queues and is value-preserving.
