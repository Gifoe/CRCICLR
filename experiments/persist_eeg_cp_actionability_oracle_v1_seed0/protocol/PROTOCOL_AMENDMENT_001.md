# Protocol amendment 001

Amended: 2026-09-25 04:36:09 UTC  
Parent lock SHA-256: `5f94135c4fb17655e2e98466f79981d50c87ae0f8e7f7330546c32384a9f2f99`  
Amended lock SHA-256: `646d26f15a5544bb748419fc6828001c001daf924865079e21d3981f931f37cd`

## Reason

The first WBCIC_MI fold 0 attempt failed its successor-C re-projection guard with absolute error `3.0517578125e-05`. Re-running the same cell on CPU produced the identical value, while the alpha=1 logit identity error remained below `4e-07`. This isolates a float32 scale/rounding guard failure, rather than a hardware-dependent result or a changed prediction.

## Change

Only the successor-C numerical acceptance check changes. For reconstructed representation `y_tilde`, mean `mu_d`, and native complement `c_d_native`, the check is now

`absolute_error < max(1e-5, 64 * float32_eps * max(1, max_abs(y_tilde), max_abs(mu_d), max_abs(c_d_native)))`.

The alpha=1 logit identity and prediction checks remain at `1e-5`. The frozen EEGNet and weights, split roles, intervention equation, alpha grid, direction ranking, oracle definitions, router gate, and bootstrap are unchanged. The check still fails if the C error exceeds a scale-aware float32 rounding bound; per-cell outputs continue to record the absolute error.

## Previously completed cells

The 15 completed cells passed the parent lock's stricter absolute `1e-5` complement check. Their curve and metric files are retained byte-for-byte only after their recorded SHA-256 values are revalidated. Their completion markers are re-attested to the amended lock without changing result files. The incomplete parent-lock WBCIC fold 0 files are discarded by the amended rerun; all five WBCIC folds use the amended lock.
