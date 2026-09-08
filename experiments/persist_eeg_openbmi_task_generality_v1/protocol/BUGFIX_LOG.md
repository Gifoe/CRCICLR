# Bug-fix log

- Before audit completion, the BIDS-sidecar fetcher was made retryable and restart-cacheable after a transport timeout. It still reads the same version-pinned official sidecars, does not alter labels, cache content, epoch definitions, splits, models, or outcomes.

- Seed-0 runner initially looked for a top-level `seeds` field although the already-committed amendment stores it under `scientific_changes`. It failed before loading data or training; the schema lookup was corrected and the amendment hash was refreshed. No protocol value or scientific outcome changed.
