# PERSIST-EEG Experiment 4 V2 — Decision-Grounded Utility-Preserving Adaptation

This is a new experiment directory. It does not overwrite the historical
`persist_eeg_exp4_protection_first_final` result. The runner first audits the
deployment alignment, rediscovers a small direction-level persistence pool in
the exact S1-only-anchor setting, and performs the utility-collapse headroom
gate before fitting any Guard.

The development runner is:

```text
python code/run_exp4_v2.py audit
python code/run_exp4_v2.py prepare
python code/run_exp4_v2.py discover
python code/run_exp4_v2.py generic
python code/run_exp4_v2.py dev
```

`outer` is fail-closed unless a separately generated final protocol lock
authorizes one one-time evaluation. Raw EEG, caches, and checkpoints are
execution-server artifacts and are not committed to Git.
