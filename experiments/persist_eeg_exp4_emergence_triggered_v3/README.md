# PERSIST-EEG Experiment 4 V3 — Emergence-Triggered Utility Preservation

V3 is a new experiment and does not overwrite Exp4 V1 or V2. It repairs the
decision-dependence centering asymmetry, calibrates random interventions by
removed representation energy, tests deployment-matched rank-1/2/4 subspaces,
and audits whether protected utility emerges during S2 adaptation. A Guard is
only trained if the predeclared emergence and subsequent-collapse gates pass.

The sealed outer cohort is fail-closed throughout development.

```text
python code/run_exp4_v3.py audit
python code/run_exp4_v3.py prepare
python code/run_exp4_v3.py certify
python code/run_exp4_v3.py trajectory
python code/run_exp4_v3.py finalize
```
