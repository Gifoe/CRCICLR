# Optimization conflict audit

Frozen PUD source checkpoints were differentiated with torch.autograd.grad on three deterministic legal source batches per fold×seed; no optimizer step was called. Parameters and buffers were hash-checked before/after. The reported cosine values are diagnostic, not a post-hoc training explanation. Existing training ledgers remain provenance for loss/validation trajectories.
