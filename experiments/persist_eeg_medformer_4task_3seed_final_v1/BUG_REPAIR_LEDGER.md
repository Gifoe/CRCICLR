# Bug repair ledger

- 2026-09-15: applied the numerically equivalent concurrent-memory repair identified in the matched ModernTCN queue. After the unchanged contiguous device transfer, redundant NumPy signal arrays are released instead of being retained alongside CUDA tensors. Dataset values, ordering, model computation, optimization, checkpoint selection, and metrics are unchanged.
