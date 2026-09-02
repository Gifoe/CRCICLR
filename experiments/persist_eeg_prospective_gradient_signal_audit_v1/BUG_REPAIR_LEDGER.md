# Engineering-only repair ledger

The audit uses immutable NumPy metadata columns and the PMG-fast mmap batch helper to avoid pandas advanced-index instability. Cache lookup has an explicit server fallback, and anchor serialization uses `weights_only=False` for the recorded local checkpoint schema. These are path/serialization repairs only; no scientific threshold, epsilon, dataset, fold, seed, architecture, or PMG coefficient was changed.
