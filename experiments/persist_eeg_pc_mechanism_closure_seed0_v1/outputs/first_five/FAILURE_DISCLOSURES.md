# First-five engineering and provenance disclosures

These are the five EEGNet / OpenBMI MI / seed0 fold previews only. A preview is not the full 20-cell mechanism-closure result.

- Intermediate historical checkpoints at the requested 10/25/50/75% trajectory points were unavailable. Each fold's trajectory rows therefore come from an exact-recipe baseline replay and are labelled `RETRAINED_REPLICA_TRAJECTORY`; they must not be described as the historical training path. Source-specific audits verified the historical selected/final endpoints, but not the missing intermediate snapshots.
- Outer-development measurements are evaluation-only and post-outcome descriptive. They do not support prospective prediction, model selection, or causal claims. Final-heldout data were not accessed.
- Several early engineering attempts failed or were interrupted before corrected versions completed. The corresponding logs, partials, and fail-closed JSON remain preserved on the original server; none is substituted for a successful result:
  - Fold 0: directional V1 throughput interruption, directional V3 memory interruption, mediation V1 fail-closed probe, and checkpoint-native P_t V1 missing-channel failure. Later versioned runs completed the relevant analyses.
  - Fold 1: mediation resume wrapper rejected a valid four-transition result because it expected five; directional queue V1/V2 had resource/orchestration failures; checkpoint-native epoch 49 was explicitly fail-closed. Later valid stages are reported separately; the missing/failed epoch evidence is not silently repaired or called historical.
  - Fold 3: endpoint/native V3 assumed the best-epoch entry occupied a fixed schedule position. The wrapper failed; ordering-independent V4 completed the six required outputs.
  - Fold 4: the server reboot interrupted directional/replay/endpoint tasks; atomic directional progress was resumed. Backtrace/checkpoint V2–V5 and subject/compact V2 had wrapper/path/argument errors and no scientific output; corrected V6 and V3 produced the outputs represented here.
- The project uses the amended final-layer historical canonical oblique projector while intermediate layers use TRAIN-only orthogonal projectors. This cross-layer geometric boundary is intentional and governed by the hashed amendment and analysis lock.
- Regime clustering is secondary and TRAIN-defined. The reported local context sensitivity is a finite-swap cosine, not a Jacobian. Results are associational and do not establish biological causality.

All failed-attempt evidence is retained in the original server runtime. This concise disclosure does not replace those source logs or failure JSONs.
