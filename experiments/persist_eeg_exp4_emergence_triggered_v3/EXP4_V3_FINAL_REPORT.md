# PERSIST-EEG Experiment 4 V3 — final report

Terminal state: **EXP4_V3_NO_PROTECTED_EMERGENCE**

## Explicit protocol answers

1. Generic reproduction: **yes**; Frozen BA=0.7011588445429909, Generic BA=0.7709187915742794, mean Generic−Frozen=0.06975994703128845.
2. Repaired metric: **yes**; candidate/random centering is symmetric, offset-invariant, and deterministic (`results/DECISION_METRIC_AUDIT.json`).
3. Control calibration: **ENERGY_MATCH_PASS (matched only on removed representation energy)**; no S3 outcome enters matching.
4. Individual directions passing the complete gate: **0**. Cumulative rank-1/2/4 subspaces passing: **0** (Holm correction across ranks within fold).
5. Training trajectory: **P range [0.1794, 0.3444], U range [-0.002966, -5.391e-05], D range [-0.0339, -0.009574]** over epochs 0/5/10/15/20/25; no rank has two consecutive full-gate checkpoints, so no t* exists.
6. Emergence/collapse: **no protected emergence**; utility-collapse and Guard-development phases were therefore not authorized.
7. Final Guard and matched controls: not run, because training-side Case C was not established; no control result is being presented as a method result.
8. Generic negative transfer: 8/41 subjects; no Guard rescue claim is permitted.
9. S3 use: held development S3 labels were used only for the predeclared baseline endpoint and never for rank, trigger, or method selection.
10. Outer access: **none**; `OUTER_LOCK.json` remains `OUTER_SEALED`, evaluation count 0, and no final outer lock was written.
11. Justified claim: under this EEGNet/S1→S2→unseen-S3 protocol, repaired and energy-matched measurements did not identify a prospectively protected persistent subspace. A universal claim about EEG persistence or adaptation is not justified.

```json
{
  "terminal_state": "EXP4_V3_NO_PROTECTED_EMERGENCE",
  "emergence_and_collapse_state": "EXP4_V3_NO_PROTECTED_EMERGENCE",
  "collapse": [
    {
      "fold": 0,
      "rank": null,
      "emerged": false,
      "t_star": null,
      "utility_t_star": null,
      "utility_final": null,
      "utility_delta_final_minus_t_star": null,
      "collapse_replicated": false,
      "generic_delta_BA": 0.1922222222222222
    },
    {
      "fold": 1,
      "rank": null,
      "emerged": false,
      "t_star": null,
      "utility_t_star": null,
      "utility_final": null,
      "utility_delta_final_minus_t_star": null,
      "collapse_replicated": false,
      "generic_delta_BA": 0.12124999999999991
    },
    {
      "fold": 2,
      "rank": null,
      "emerged": false,
      "t_star": null,
      "utility_t_star": null,
      "utility_final": null,
      "utility_delta_final_minus_t_star": null,
      "collapse_replicated": false,
      "generic_delta_BA": 0.004374999999999974
    },
    {
      "fold": 3,
      "rank": null,
      "emerged": false,
      "t_star": null,
      "utility_t_star": null,
      "utility_final": null,
      "utility_delta_final_minus_t_star": null,
      "collapse_replicated": false,
      "generic_delta_BA": 0.004394728535353462
    },
    {
      "fold": 4,
      "rank": null,
      "emerged": false,
      "t_star": null,
      "utility_t_star": null,
      "utility_final": null,
      "utility_delta_final_minus_t_star": null,
      "collapse_replicated": false,
      "generic_delta_BA": 0.011249999999999987
    }
  ],
  "summary": [
    {
      "method": "Frozen",
      "n_subjects": 41,
      "BA_mean": 0.7011588445429909,
      "delta_BA_vs_Frozen_mean": 0.0,
      "negative_transfer_rate": 0.0,
      "negative_transfer_count": 0,
      "worst_quartile_delta": 0.0,
      "worst_subject_delta": 0.0
    },
    {
      "method": "Generic",
      "n_subjects": 41,
      "BA_mean": 0.7709187915742794,
      "delta_BA_vs_Frozen_mean": 0.06975994703128845,
      "negative_transfer_rate": 0.1951219512195122,
      "negative_transfer_count": 8,
      "worst_quartile_delta": -0.004985651974288402,
      "worst_subject_delta": -0.0150000000000001
    }
  ],
  "outer_accessed": false,
  "outer_authorized": false
}
```

Outer subjects were not accessed during development.
