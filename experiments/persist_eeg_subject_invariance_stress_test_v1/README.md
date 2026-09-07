# PERSIST-EEG subject-invariance stress test V1

This experiment audits, rather than assumes, the link between source-learned subject invariance and unseen-subject future-session performance. It compares matched ERM, DANN/GRL, multi-domain CORAL, and multi-domain MMD under the frozen five-fold, three-seed OpenBMI V8_SEARCH protocol on EEGNet and EEGConformer.

The fixed scientific contract is `STRESS_TEST_PROTOCOL_FROZEN.json`. All method strengths are declared before outcome evaluation. Model selection and lambda selection use only subject-disjoint source validation. Outcome Session-2 labels are read only after the unit-level selection artifact has been frozen.

Server entry points:

```text
python code/preflight.py
python code/run_stress.py --backbone eegnet --fold 0 --seed 0
python code/scheduler.py
python code/aggregate.py
python code/validate.py
```

Runtime checkpoints, embeddings, and logs remain untracked. Lightweight tables, statistics, figures, and audit reports are committed.
