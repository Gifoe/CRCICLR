# PERSIST-EEG Decision-Relevant Prospective Harm Audit V1

EEGNet, seed 0, OpenBMI/WBCIC folds 0--4. This is a source/refit-only audit of the frozen TASK_ONLY_MATCHED trajectory. It does not train a guard and does not open development outcome, WBCIC outer-10, or OpenBMI sealed/confirmation data.

The primary surrogate is frozen class-balanced Balanced Boundary Risk (BBR); CE is a matched comparator. Exact decision harm is held-out B_out Balanced Error harm and correct-to-wrong flips within the same legal source/refit subjects.

The machine-readable pre-outcome lock is `results/PRE_OUTCOME_LOCK.json`; scientific interpretation is in `FINAL_REPORT.md` and `AUTONOMOUS_DECISION.md`.
