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
