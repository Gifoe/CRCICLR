# V3 ICLR readiness assessment

Decision: **NO-GO**.

Failed criteria:

- paired subject-bootstrap set-size CI does not remain above zero
- Probe-policy intervention rate is zero on at least one primary task
- positive utility is not replicated across seeds

Oracle GO was necessary but not sufficient. It cannot override failed deployable-policy evidence.

## Answers to the twelve predeclared questions

1. Safe-Oracle headroom exists, but is very small: mean relative reduction by dataset is `{'eegmmidb': 0.0020257314834651, 'hmc': 0.0038120165302216}`; T3A contribution is zero.
2. A/P versus same-context screening is not established: the required `no_adapt_probe_split` surfaces were not generated, so no causal comparison is claimed.
3. Probe diagnostics have weak and dataset-dependent association with future gain; the full development-only Spearman table is in `V3_NESTED_DEVELOPMENT_REPORT.md`. No diagnostic is stable enough to support a general claim.
4. ProbeCert does not consistently beat No-TTA+CRC at matched protocol; paired mean set-size reduction (positive favors ProbeCert) is `{'eegmmidb': 0.00020627062706265, 'hmc': -0.28011702674897115}`.
5. The observed result cannot be attributed robustly to intervention: mean intervention rate is 0.0801; seed-positive counts are `{'eegmmidb': 0, 'hmc': 2}`.
6. Mean intervention rate across primary summaries is 0.0801.
7. Estimated selected-intervention non-harm PPV is 0.8921; it is undefined when a stratum selects no interventions.
8. Mean full-set/sentinel rate is 0.1447; HMC remains substantially sentinel-dominated at alpha=0.1.
9. Calibration-size sensitivity is in `CALIBRATION_SIZE_SENSITIVITY.csv`; small m frequently forces the finite-sample sentinel, as predicted by the order statistic.
10. Mean set size for actionwise simultaneous versus policy-level certificate is `{'eegmmidb': 4.0, 'hmc': 5.0}` versus `{'eegmmidb': 3.592525828460039, 'hmc': 3.582560606060606}`; lower is less conservative.
11. HMC, EEGMMIDB, and CAP do not provide a consistent positive conclusion; CAP is replication evidence only and was not used as untouched confirmation.
12. The current evidence is insufficient for a new untouched confirmation dataset. The decision is NO-GO.
