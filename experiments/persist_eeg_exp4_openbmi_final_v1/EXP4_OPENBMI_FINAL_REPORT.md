# PERSIST-EEG Experiment 4 — OpenBMI MI closure

## Terminal state

**EXP4_OPENBMI_MECHANISM_NOT_ACTIONABLE**

The descriptive mechanism is measurable, but the complete empirical chain does not close. The P/U/D guard is not prospective or control-specific on V8_SEARCH, and no sealed confirmation was opened.

## Protocol and legality

1. OpenBMI MI is primary because Exp1–Exp3 causal and decision-grounding evidence was established on the same resource.
2. Deployment is history-to-future: target Session 1 labels adapt; target Session 2 is unseen outcome.
3. Development used only V8_SEARCH (40 subjects; five subject-only folds).
4. V8 internal holdout remained sealed (14 subjects); historical outer-test remained sealed.
5. No target Session-2 labels, embeddings, predictions or partial metrics were used during search.

## Baselines

6. MI-specific NoAdapt=0.8812; MI-specific Generic=0.8822.
7. Strongest fair Generic audit: Conformer-Norm=0.8947, negative-transfer rate=0.325. It is stronger than the old 0.83778 historical reference, though cohort sizes differ.
8. MI-specific Generic harm rate=0.400; it harms a substantial development subset.

## Mechanism and intervention

9. MI-specific PUD subspaces show persistence strength 0.637–0.804 and positive signed utility in source folds.
10. Symmetric centered decision dependence is finite and invariant to additive class-logit shifts.
11. DGUG_PROTECT=0.8845 (+0.0022 vs MI Generic; harm rate=0.325). Utility trust-region=0.8850; utility-only=0.8858; decision-only=0.8880; random matched=0.8832.
12. Decision-only and utility-only controls are at least as strong as the PUD guard; history-side predictor correlations are weak. Therefore Decision Grounding adds no demonstrated unique actionability.
13. Three MI seeds reproduced the same closed-form direction; this is robustness of the negative specificity finding, not confirmation.

## Confirmation boundary

14. `OPENBMI_EXP4_FINAL_LOCK.json` was not created. Internal holdout and outer-test results are intentionally absent.
15. No second-backbone method search was used to rescue the result; Conformer-Norm was audited only for Generic strength.
16. The strongest justified claim is an actionability boundary: persistent and causally useful decision-responsive structure exists descriptively, but the current P/U/D intervention does not prospectively improve future-session adaptation beyond simpler controls.
17. The stronger claim that Persistence→Causal Utility→Decision Grounding→Better Future Adaptation is a complete empirical chain is not justified.

Runtime artifacts and failed variants are in `results/`, `protocol/`, `figures/`, and `ITERATION_LEDGER.md`.
