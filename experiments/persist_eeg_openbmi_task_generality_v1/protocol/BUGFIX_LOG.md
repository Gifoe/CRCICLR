# Bug-fix log

- Before audit completion, the BIDS-sidecar fetcher was made retryable and restart-cacheable after a transport timeout. It still reads the same version-pinned official sidecars, does not alter labels, cache content, epoch definitions, splits, models, or outcomes.
