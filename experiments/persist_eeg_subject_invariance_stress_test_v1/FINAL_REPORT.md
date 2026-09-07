# PERSIST-EEG subject-invariance stress test V1 — final report

Exact terminal state: **SUBJECT_INVARIANCE_AUDIT_STRONG_MISALIGNMENT_SUPPORTED**.

## Required answers

1. Did DANN reduce cross-session subject identifiability? Yes for 1 backbone-strength cells; largest was eegconformer lambda=0.1, mean suppression=0.050209, CI=[0.04096418107406743, 0.0601491424772442].
2. Did CORAL reduce it? No predeclared meaningful reduction; largest mean suppression=0.012670.
3. Did MMD reduce it? Yes for 2 backbone-strength cells; largest was eegconformer lambda=1, mean suppression=0.068065, CI=[0.058696499735195265, 0.07720903324657678].
4. Did BA reliably rise when identity fell? No reliable global positive alignment was established.
5. Cluster-aware slope: 0.32713343, CI=[-0.18614555463457355, 0.7453209075561887].
6. Clean identity-down/BA-down-or-null configurations: 3; all configurations are reported rather than post-selected.
7. EEGNet qualitative counterexample evidence: yes.
8. EEGConformer qualitative counterexample evidence: yes.
9. Decision versus identity: RMSE(MI)-RMSE(MD)=0.00163586, CI=[0.000363625540601402, 0.0027646249714080467].
10. Persistence: direction-level values are retained in `results/direction_audit.csv`; no utility implication is assumed.
11. Generic adaptation: not rerun; it remains an optional separate reference and is excluded from source-only correlation.
12. Restricted access: none; only the exact authorized 40-subject cache was loaded.
13. Terminal state: `SUBJECT_INVARIANCE_AUDIT_STRONG_MISALIGNMENT_SUPPORTED`.
14. Strongest defensible claim: Reducing subject identifiability was not a reliable operational target for future-session generalization in this fixed OpenBMI stress test; multiple standard invariance families produced identity reduction without consistent BA gain, while finite decision dependence was a more task-grounded consequence predictor.

This is an operational representation audit, not a biological causal claim. It does not establish that identity never matters, that invariance is universally harmful, or that decision dependence guarantees generalization.
