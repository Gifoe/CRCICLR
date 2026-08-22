# PERSIST-EEG Experiment 3 V1.2 scientific report

This is a development-resource measurement repair and causal closure on the reused development resource; it is not an untouched confirmation or independent replication. V1 and V1.1 remain unchanged.

## Required questions

1. **Why did V1.1 G2 fail?** Top-1 subject-ID BA was discrete and the 21-point grid selected alpha=0 for most Protected MEDIUM interventions; held-out P identity reduction was therefore exactly zero.

2. **Primary continuous metric:** `symmetric_cross_session_subject_id_identity_skill_raw` with IdentitySkill=log(K)−cross-session subject-ID CE, symmetric S1→S2/S2→S1.

3. **Why this metric / validation tuning?** Log-loss uses probabilistic identity evidence and has finer resolution than top-1 BA. The metric was selected by the predeclared hierarchy and train-only numerical audit; validation task outcomes were not used.

4. **Top-1 vs continuous train curve:** top-1 BA is stepwise/saturated, while log-loss skill varies continuously under suppression; the comparison is in `IDENTITY_METRIC_COMPARISON.csv` and Figure 5.

5. **Protected continuous measurability:** 0/10 blocks (0.000) passed the train noise-floor gate.

6. **Measurable P interventions across runs:** 0/6 eligible runs under the frozen train-only gate.

7. **MEDIUM eligibility:** 0/10 blocks entered MEDIUM; fixed V1.1 controls only.

8. **Fixed N controls:** train-only MEDIUM eligible count is recorded per block in `IDENTITY_MATCHING_PARAMETERS.csv`; minimum=0.

9. **MEDIUM train target:** per-block values are in `IDENTITY_TARGETS.csv`; target is 50% of train Dmax_P and must exceed the frozen noise floor.

10. **Held-out continuous reductions:** ΔID_P=None; ΔID_N=None; difference=None.

11. **G2 equivalence:** epsilon=None; 90% CI=[None, None]; equivalence=False; G2=False.

12. **Top-1 direction agreement:** reported in `SECONDARY_IDENTITY_METRICS.csv`; it is secondary and does not override the continuous metric.

13. **Task outcome:** not run because G1 train-only design was infeasible.

14. **ΔH robustness:** not applicable because G1 train-only design was infeasible.

15. **Dose response:** {}.

16. **Random controls:** secondary only; see `RANDOM_CONTROL_DIAGNOSTICS.csv`; they never affect fixed matching or gates.

17. **Secondary identity metrics:** CE, top-1 BA, correct-subject margin and retrieval margin are all retained; any disagreement is reported as metric dependence rather than hidden.

18. **Metric dependence:** the primary decision is based only on frozen log-loss identity skill; secondary metrics cannot rescue or overturn G2.

19. **Validation-outcome tuning:** none. Matching membership, metric, noise floor, doses, epsilon and gates were frozen before new held-out evaluation.

20. **Outer:** untouched; all artifacts set both outer flags false.

21. **Final label:** `NOT_IDENTIFIABLE`; Theory 3 state: `NOT_IDENTIFIABLE`. Train-only continuous identity manipulation was not feasible under the frozen protected-block and fixed-control design.

22. **READY_FOR_EXPERIMENT_4:** `NO`.

## Decision

Terminal: `PROTECTED_PERSISTENCE_NOT_IDENTITY_BEARING_UNDER_OPERATIONAL_METRICS`; G0=False; G1=False; G2=False; G3=None.
