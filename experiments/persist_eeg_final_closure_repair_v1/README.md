# PERSIST-EEG final closure repair v1

This experiment repairs the final PUD constructive closure without changing the scientific question, P/U/D definitions, folds, seeds, lambda grid, or sealed-data policy.

Phase A is evaluation-only and repairs the B0 subject join, the PUD certificate-direction join, the B3 bottleneck gate, and hierarchical certificate-transfer inference. Phase B is the only training stage and compares PUD-Aux against a newly trained exactly matched lambda-zero task-only pipeline under strict nested source selection.

Run with the GPU environment on the server:

```text
python code/preflight.py
python code/phase_a_repair.py
python code/matched_aux.py --fold F --seed S
python code/matched_aux.py --aggregate
```

The internal 14-subject OpenBMI holdout and WBCIC outer test are forbidden. Runtime caches and checkpoints are excluded from Git.

