# PERSIST-EEG V2.1 falsification audit

## Decision

`ENSEMBLE_EXPLAINS_V2_GAIN`

Secondary qualifiers: `PROTECTED_SAFE_PREFERRED, DEPLOYMENT_DEFINITION_REQUIRED`.

This is a post-V2 exploratory disambiguation analysis. The prior 12-subject
development holdout had already been opened; it is not presented as a new
confirmatory holdout. The 40-subject exploration pool and 12-subject holdout
remain separate. The pooled 52-subject rows are descriptive only. WBCIC outer
was not accessed.

## Direct answers

1. Exact V2 reconstruction: **True**. The frozen policy lock,
   four source-cache hashes, every persisted subject/run/action/summary value,
   and the historical exploration ORACLE summary passed at tolerance `1e-14`.
2. FULL vs C2 hard-label equivalence: **True**.
3. Protected-safe vs C3 hard-label equivalence: **True**.
4. Other-run hard majority vs target KEEP on the existing holdout:
   **+0.882 pp**.
5. All-run hard majority vs target KEEP:
   **+0.903 pp**.
6. Probability averaging: other-run
   **+0.885 pp**;
   all-run **+1.986 pp**.
7. Logit averaging: other-run
   **+0.844 pp**;
   all-run **+2.028 pp**.
8. Strongest predefined KEEP-only ensemble, selected using exploration only:
   `B6_ALL_RUN_LOGIT_MEAN`.
9. FULL minus best ensemble: **-1.181 pp**,
   subject-bootstrap CI95 [-1.979 pp, -0.438 pp].
10. Protected-safe minus best ensemble:
    **-1.295 pp**,
    subject-bootstrap CI95 [-2.049 pp, -0.590 pp].
11. The CIs above use subjects, not 9,200 target-run rows, as replicates.
12. Intervention-specific hard-label improvement: **False**.
    For Balanced Accuracy / Accuracy / F1, the intervention policy is
    prediction-equivalent to an action-masked consensus override when C2/C3
    agreement is 100%.
13. Probabilistic benefit under the frozen criterion: **False**.
    Comparator: `B7_CONFIDENCE_WEIGHTED_KEEP_ENSEMBLE`. This is reported
    separately and does not alter the classification conclusion.
14. Most unique action rescue beyond the best ensemble:
    `ERASE_MATCHES_MAJORITY`
    (55 target-run rows).
15. ERASE selected 443 rows, rescued
    232, harmed 211,
    and had net correctness +21; unique rescue
    beyond the best ensemble was 55.
16. Protected-safe preferred as a secondary safety qualifier:
    **True**.
17. Negative FULL runs: `fold-0_seed-0, fold-1_seed-0`.
    2/2
    become nonnegative when ERASE is forbidden; ERASE is the only structural
    difference between FULL and protected-safe.

    Run-specific ERASE evidence: `[{"run":"fold-0_seed-0","full_delta_BA":-0.007500000000000062,"safe_delta_BA":0.0012499999999999734,"best_ensemble_delta_BA":0.011874999999999969,"erase_selected":86,"erase_rescue":36,"erase_harm":50,"erase_net_correctness":-14},{"run":"fold-1_seed-0","full_delta_BA":-0.0031249999999999334,"safe_delta_BA":0.009375000000000022,"best_ensemble_delta_BA":0.0031250000000000444,"erase_selected":72,"erase_rescue":26,"erase_harm":46,"erase_net_correctness":-20}]`.
18. A unique deployed I003 trial prediction is defined: **False**.
    Current status: `DEPLOYMENT_OUTPUT_NOT_YET_DEFINED`.
19. Supported mechanism: **generic KEEP-only ensemble gain**.
20. Next experiment: Freeze one unique trial-level deployment rule before outcomes, then compare compute-matched all-run KEEP ensembles against action-masked direct consensus and real interventions on genuinely new subjects or a new dataset. Use subject-level inference and authorize any outer set only under a separate protocol.

## Scientific interpretation

V2 found a usable cross-run consensus signal. It did not, by itself, establish
that AMPLIFY, GEOMETRY, or ERASE creates the hard-label gain. A binary action
that flips the target prediction toward the opposite leave-target-run majority
necessarily emits the same class as direct consensus. The mandatory C2/C3
controls quantify that identity rather than treating intervention semantics as
evidence.

All inference comparisons are compute-disclosed in
`ENSEMBLE_BASELINE_RESULTS.csv` and `CONSENSUS_CONTROL_RESULTS.csv`. A
multi-model intervention policy is not compared only with a single model.

`OUTER_TEST_USED = false`
