# Engineering repair log

- Added a predicate-pushed, role-scoped label loader and subject-scoped signal materialization to enforce the two-phase boundary.
- Split Phase-2 training/evaluation into resumable source and outcome processes with a 30-run SHA-256 global freeze barrier.
- Replaced a slow pandas-loop hierarchical bootstrap with an algebraically equivalent vectorized resampler after a runtime bottleneck; retained 10,000 draws, hierarchy, direction resampling/re-ranking, and frozen seeds. The interrupted partial aggregate produced no final statistics and was invalidated.
- Vectorized the 1,000 fixed within-run permutation policies and used the same frozen leave-run ridge definition for prediction nulls.
- Kept architectures, preprocessing, ERM optimizer recipe, direction family, intervention, statistics, and frozen gates unchanged.
