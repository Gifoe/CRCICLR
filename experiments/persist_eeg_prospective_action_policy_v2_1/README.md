# PERSIST-EEG prospective action policy V2.1

V2.1 is a fixed, post-V2 falsification audit. It asks whether the positive V2
classification result is intervention-specific or is explained by ordinary
multi-run KEEP ensembling and consensus.

It does not train a new router, tune a threshold, reopen a sealed holdout, or
access WBCIC outer data. The historical 40-subject exploration pool and the
already-opened 12-subject development holdout are reported separately.

The finite baseline family (B0--B7), direct-consensus controls (C1--C3),
confidence weighting, thresholds, tie rule, bootstrap unit, and ensemble
selection rule are frozen in `outputs/protocol/V2_1_ANALYSIS_SPEC.json` before
outcome comparison. The best KEEP-only ensemble is selected on the original
40-subject exploration pool only and then applied unchanged to the already
opened 12-subject development holdout.

Run the exact reconstruction gate before the audit:

```powershell
python experiments/persist_eeg_prospective_action_policy_v2_1/code/run_all.py --phase reconstruct
python experiments/persist_eeg_prospective_action_policy_v2_1/code/run_all.py --phase test
python experiments/persist_eeg_prospective_action_policy_v2_1/code/run_all.py --phase evaluate
```

Or run the complete fixed pipeline:

```powershell
python experiments/persist_eeg_prospective_action_policy_v2_1/code/run_all.py --phase all
```

The mandatory sample-level comparisons are FULL vs C2 and protected-safe vs
C3. A 100% match means BA, accuracy, and macro-F1 are prediction-equivalent;
probability/calibration results are reported separately and cannot be used to
relabel the hard-classification mechanism as intervention-specific.

WBCIC outer remains unauthorized and unused.

## Frozen server result (2026-08-19)

The server execution passed exact V2 reconstruction and all six unit tests.
The exploration-only selection chose `B6_ALL_RUN_LOGIT_MEAN`. On the existing
12-subject development holdout:

- B6 minus target KEEP: `+2.028 pp`, subject-bootstrap CI95
  `[+1.115, +3.014] pp`;
- frozen FULL minus B6: `-1.181 pp`, CI95 `[-1.979, -0.438] pp`;
- frozen protected-safe minus B6: `-1.295 pp`, CI95
  `[-2.049, -0.590] pp`;
- C2 vs FULL and C3 vs protected-safe: zero hard-prediction disagreements
  across all 9,200 holdout target-run rows.

The primary state is therefore `ENSEMBLE_EXPLAINS_V2_GAIN`. The result is
negative for intervention-specific hard-label value: ordinary all-run KEEP
logit averaging is materially stronger. Protected-safe remains a secondary
safety qualifier because it removes ERASE, lowers harm, and converts both
negative FULL target runs to nonnegative, but it does not beat B6.

See `outputs/SCIENTIFIC_REPORT.md` and `outputs/FINAL_DECISION.json`. The
server runtime and hashes are recorded in `outputs/REPRODUCIBILITY.json`.
