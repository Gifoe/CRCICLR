# PERSIST-EEG prospective action policy V2

This development-only experiment tests whether the OpenBMI sample-level
intervention oracle can be recovered prospectively. It uses a deterministic
subject split, nested subject-grouped exploration, an immutable policy lock,
and a single opening of the development holdout.

The WBCIC outer test is outside this experiment and is never loaded.

Run the three phases separately:

```powershell
python experiments/persist_eeg_prospective_action_policy_v2/code/run_all.py --phase explore
python experiments/persist_eeg_prospective_action_policy_v2/code/run_all.py --phase freeze
python experiments/persist_eeg_prospective_action_policy_v2/code/run_all.py --phase holdout
```

Set `PERSIST_ROUTER_CACHE_ROOT` when the historical router cache is not at the
server default recorded in `code/common.py`.

