# PERSIST-EEG XS counterfactual factorial audit v1

Development-only frozen-checkpoint analysis of Channel gate (`C`), Scale gate (`S`), and residual temporal Mixers (`M`) across the complete 2^3 cube.

Run in order:

```bash
python code/replay_counterfactual_cube.py --repo "$REPO" --runtime "$RUNTIME"
python code/compute_rescue_harm.py --repo "$REPO"
python code/backward_ablation_transitions.py --repo "$REPO"
python code/factorial_effects.py --repo "$REPO"
python code/aggregate_counterfactual_audit.py --repo "$REPO"
```

No EEG network is trained. No internal heldout, final heldout, final test, or sealed test artifact is accessed.

`COUNTERFACTUAL_TRIAL_RESULTS.csv` uses a documented lossless compact encoding (`OMI`, `OERP`, `OSSVEP`, `WMI`; correctness as 0/1; WMI subject numeric suffix). The analysis scripts restore canonical task and subject names. This keeps the required CSV below GitHub's single-blob limit when LFS storage is unavailable.
