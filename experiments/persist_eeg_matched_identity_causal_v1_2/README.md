# PERSIST-EEG Experiment 3 V1.2

V1.2 is the final pre-specified continuous identity measurement repair. It
preserves V1.1 and reuses its frozen Protected blocks and matched control
membership without re-selection. The only changed quantities are the primary
continuous identity metric, train-only noise-floor calibration, dense alpha
solving, and dose eligibility.

Run on the server:

```text
python experiments/persist_eeg_matched_identity_causal_v1_2/code/experiment3.py phase0
python experiments/persist_eeg_matched_identity_causal_v1_2/code/experiment3.py train_only
python experiments/persist_eeg_matched_identity_causal_v1_2/code/experiment3.py freeze
python experiments/persist_eeg_matched_identity_causal_v1_2/code/experiment3.py final_identity
python experiments/persist_eeg_matched_identity_causal_v1_2/code/experiment3.py finalize
```

Only if `G2_pass=true` in `outputs/G2_EQUIVALENCE_TEST.json` may the task phase
be run:

```text
python experiments/persist_eeg_matched_identity_causal_v1_2/code/experiment3.py final_task
python experiments/persist_eeg_matched_identity_causal_v1_2/code/experiment3.py finalize
```

The development validation subjects were already inspected in V1/V1.1, so this
is not an untouched replication. Outer data remain sealed. A failed G2 is a
scientific `NOT_IDENTIFIABLE` closure, not a reason to create V1.3.
