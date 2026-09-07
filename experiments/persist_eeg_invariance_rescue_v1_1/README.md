# PERSIST-EEG Invariance Rescue V1.1

This is an exploratory redesign after the completed V1 pilot (`e96e3907`).
V1 is preserved unchanged. V1.1 repairs the measurement by pairing a task-only
anchor and invariant model, adding an independent task-only replica, selecting
GRL strength on train-subject inner CV, refitting on all 34 legal train
subjects, and measuring teacher-defined task evidence rather than raw latent
coordinate R². Full evaluation is performed on the nine frozen development
outcome subjects, Session 2. The outer split is sealed.

Run phases with `python code/run.py phase0`, `smoke`, `freeze`, `full`,
`audit`, `functional`, `rescue`, and `finalize` in that order. Full phases are
authorized only on the designated CUDA server.
