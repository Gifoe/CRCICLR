# PERSIST-EEG residual actionability V3

V3 is an exploratory residual-headroom audit above the frozen V2.1 reference
`B6_ALL_RUN_LOGIT_MEAN`. It does not reinterpret the V2 gain and does not
access WBCIC outer subjects.

The mandatory first stage reconstructs V2.1, collapses target-run duplicates
to one row per EEG trial, constructs the four predeclared action menus and the
finite KEEP-only diversity control, and applies the frozen headroom gate.
Policy learning is prohibited unless that gate returns
`STRUCTURAL_ACTION_RESIDUAL_EXISTS`.

The pre-outcome M1-M5 run was frozen before its grouped OOF outcomes. Its
conditional audit found that unconditional learnability was inflated by rows
where an action could not change B6. One finite adaptive follow-up (I006-I007)
therefore trains action-specific rescue-vs-harm heads only on boundary-cross
candidates. It uses the same outer folds, calibrates a finite KEEP/FULL/
protected-safe/ERASE-only menu on inner subjects, and is explicitly
post-primary-adaptive exploratory evidence. No MLP or unrestricted search is
run.

Server execution:

```powershell
python experiments/persist_eeg_residual_actionability_v3/code/run_all.py --phase diagnose
python experiments/persist_eeg_residual_actionability_v3/code/run_all.py --phase all
```

All 52 subjects are historical development data. Every result is exploratory.
`OUTER_TEST_USED=false` throughout.
