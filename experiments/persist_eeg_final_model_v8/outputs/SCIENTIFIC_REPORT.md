# PERSIST-EEG V8 scientific report

## Decision

`V8_SCIENTIFIC_EXHAUSTION_PHASE_A_HEADROOM`

This is a negative result.  The broadest competence-filtered bank remained
below the hard gate on both benchmarks.  Phase B/C, multi-seed confirmation,
internal holdout, and WBCIC outer evaluation were correctly skipped.

## Required questions

1. **Authoritative V7 limits.** OpenBMI strongest generic was
   0.837778 BA with
   1.481 pp V7 oracle headroom. WBCIC was
   0.824967 BA with
   0.964 pp.
2. **New families.** Low-rank coverage, Meta-SGD, metric/prototype transport,
   norm/FiLM hypernetwork, SPD transport, raw fine-tuning, multi-scale TCN,
   competence-first training, low-rank expert adapters, raw FOMAML, and a
   multi-backbone union were evaluated.
3. **Families creating material headroom.** None reached +4 pp as a complete
   two-fold family.  The broad union was also below +4 pp.
4. **Strongest single expert on the two-fold screen.** OpenBMI:
   `RAW_ENCODER_FINETUNE_BANK__TAIL_FAST_RESIDUAL50` at 0.858889 BA. WBCIC:
   `MULTISCALE_TCN_STAGED_BANK_K4__E0_GENERIC_BLEND50` at 0.822917 BA.
5. **Strongest bank oracle.** OpenBMI 0.879444; WBCIC 0.843750.
6. **OpenBMI oracle headroom.** 2.778 pp versus the V7 two-fold baseline, but only
   2.056 pp after the mandatory screening-baseline update.
7. **WBCIC oracle headroom.** 3.000 pp versus V7 two-fold, and
   2.083 pp after update.
8. **Dual +8 pp gate.** No.
9. **At least +5 pp rescue potential versus V7 two-fold baseline.** OpenBMI
   22.2%; WBCIC 25.0%.
10. **Expert competence.** The union applied a fixed competence filter, but it
    was outcome-screened and is only an optimistic upper bound.  It cannot be
    presented as prospective evidence.
11. **Error diversity.** Mean pairwise correctness correlations were
    0.804 (OpenBMI) and
    0.704 (WBCIC); residual
    complementarity was insufficient.
12. **META-GENERIC oracle recovery.** Not run; gate failed.
13. **META-GENERIC actual gain.** Not run.
14. **PERSIST additional value.** Not evaluated in V8.
15. **PERSIST BA improvement.** No V8 estimate exists.
16. **PERSIST safety improvement.** No V8 estimate exists.
17. **P/U/D/G/R.** Definitions were reserved for learned-transform audits, but
    were not estimated because Phase C was never authorized.
18. **Suppression.** Not used; coefficient remains zero.
19. **Largest WBCIC single-candidate improvement.** Competence-first
    multi-scale TCN generic blending, +0.917 pp over the V7 two-fold baseline.
20. **Largest OpenBMI single-candidate improvement.** Raw encoder adaptation,
    +0.722 pp over the V7 two-fold baseline.
21. **Same K=1/K=2 core.** Shared representation/adaptation families ran on
    both settings; neither passed.  The later multi-scale/FOMAML repair was
    prioritized on failing WBCIC and was not promoted to full cross-benchmark
    evaluation because its WBCIC screen failed.
22. **Stronger fair baseline found.** Yes, on the exploratory two-fold screens.
23. **Baseline claims updated.** Yes.  The stricter updated-baseline headroom is
    reported separately from V7-locked comparability.
24. **Internal holdout opened after freeze.** No; no candidate qualified for a freeze.
25. **WBCIC outer untouched.** Yes: split not opened, subjects not enumerated,
    and raw/features/labels not loaded (`OUTER_TEST_USED=false`).
26. **Dual +5 pp actual gain.** No.
27. **Generic contribution.** Best observed screening gains were
    0.722 pp and 0.917 pp; these are exploratory, not final estimates.
28. **Unique PERSIST contribution.** 0 pp measured because Phase C was not run.
29. **Exact bottleneck.** Learned-action/representation headroom.  Even a broad
    outcome-only subject oracle remained about +2.1 pp above the updated
    matched baselines.
30. **Further iteration justified.** Not on these repeatedly reused development
    outcomes without a materially new representation source or independent
    development cohort.  Small architectural/hyperparameter changes would be
    additional adaptive search, not a credible new hypothesis.

## Scope limitations

Results are fixed-seed, two-fold V8_SEARCH screens.  They are not full five-fold
estimates, not multi-seed estimates, and not confirmation results.  The union
candidate filter used the same search outcomes, making its oracle deliberately
optimistic.  No confidence interval or final leaderboard claim is warranted.
