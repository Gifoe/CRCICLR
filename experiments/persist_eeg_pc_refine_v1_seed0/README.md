# PC-Refine EEGNet V1, seed 0

This experiment uses the frozen four-task five-fold SEARCH split, the canonical
EEGNet checkpoint recipe, TRAIN-only PERSIST Protected discovery, and the
formal OpenBMI 14-subject / WBCIC true-outer 10-subject final evaluator. Per the
user's narrowed scope, it trains only PROTECTED_PC_REFINE and the matched
canonical seed-0 baseline needed to measure its heldout gain.
The intermediate pathway projector uses the audited PathFit ridge alpha 1.0.
The earlier alpha-0.01 training run was archived before heldout access and
is excluded from this experiment's results; see the protocol amendment.

Run `python code/run.py preflight`, `python code/run.py discover --task TASK --fold F`,
`python code/run.py refit --task TASK --fold F`, `python code/run.py lock`,
`python code/run.py final-eval --task TASK --fold F`, then
`python code/run.py aggregate`. The final evaluator refuses to load heldout
arrays until the hashed lock exists and verifies all 20 cells and both groups.

After the primary result is complete, `python code/exploratory_random.py` runs
Random-PC only for tasks with a positive Protected-PC minus Baseline primary
heldout BA point difference. This conditional comparison is explicitly
exploratory and does not alter the locked V1 result.

Large checkpoints and trial predictions are stored in `PC_REFINE_RUNTIME` and
are deliberately excluded from Git. The compact audit and report files are
written to this experiment's `outputs` directory.
