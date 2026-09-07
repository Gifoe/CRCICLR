# PERSIST-EEG Experiment 3 — decision-grounding closure V1

This is the final Experiment-3 closure on the reused OpenBMI MI development
resource. It does not reopen the V1/V1.1/V1.2 matched-identity intervention,
does not change Signed-V3.1 Protected assignments, and does not open outer
subjects.

The closure asks whether frozen decision dependence predicts the already frozen
task consequence better than cross-session subject identity evidence. The
primary identity measure remains the V1.2 symmetric cross-session
`log(K)-CE` identity skill. Identity suppression is not required here: for a
cell, identity evidence is the full skill minus the skill after erasing the
frozen block, measured on that DDA cell's fit subjects.

On the server, after the source artifacts and V1.2 feature caches are present:

```powershell
$env:PERSIST_DDA_ROOT = 'D:\nips-temp\TotalP\P1\CRCICLR_INVARIANCE_RESCUE_V1\experiments\persist_eeg_dda_v1'
$env:PERSIST_V12_ROOT = 'D:\nips-temp\TotalP\P1\CRCICLR_EXP3_MATCHED_CAUSAL_V1_2\persist_eeg_matched_identity_causal_v1_2'
$env:PERSIST_EXP3_GIT_COMMIT = '668d0fca06bd1756c935f1997945fc419c391dc0'
python experiments\persist_eeg_exp3_decision_grounding_closure_v1\code\run_closure.py audit
python experiments\persist_eeg_exp3_decision_grounding_closure_v1\code\run_closure.py compute
python experiments\persist_eeg_exp3_decision_grounding_closure_v1\code\run_closure.py finalize
```

The compact outputs are under `results/`; feature caches and raw EEG remain
outside this experiment directory and are never committed.
