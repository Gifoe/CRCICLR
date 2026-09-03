# Bug repair ledger

The V2 implementation uses only engineering-level repairs: vectorized canonical batch access, exact AdamW parameter-step capture, post-step displacement restoration with optimizer moments retained, scalar correction masks, deterministic schedule serialization, BN freeze assertions and compact summaries. No candidate, kappa, LR, scope or success rule was changed after outcomes.
