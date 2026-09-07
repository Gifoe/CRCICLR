# PERSIST-CF

PERSIST-CF tests whether empirical subject offsets inside the frozen
Signed-V3.1 Protected coordinates can be varied during training without
altering the TRAIN-derived shared MI geometry.  It is a new experiment and
does not modify or reinterpret P1--P6 or PERSIST-Router.

The implementation is deliberately fail-closed:

- OpenBMI outer-test samples and labels are never loaded.
- Development-validation rows are not materialised before
  `outputs/locked/PERSIST_CF_LOCK.json` exists.
- Every inner-CV geometry, subject center, donor distribution, scale and
  stress bank is estimated from inner-training subjects only.
- Counterfactuals are injected at the frozen historical `h0` representation
  and then passed through the same P5.1 V2 adapter/head architecture.
- The exact Signed-V3.1 canonical transform and residual-preserving
  coordinate reconstruction are reused.

Run from the repository root:

```powershell
python experiments/persist_eeg_cf/code/persist_cf.py audit --device cuda
python experiments/persist_eeg_cf/code/persist_cf.py cf0 --device cuda
python experiments/persist_eeg_cf/code/persist_cf.py cf1-hard --device cuda
python experiments/persist_eeg_cf/code/persist_cf.py finalize
```

`audit` must pass both the augmentable-offset and counterfactual-validity
gates before any model training command is accepted.  In the completed run,
CF0 and the evidence-authorized CF1-HARD refinement both failed the frozen
continuation gates.  Consequently no method lock or development evaluation is
permitted; `outputs/final/PERSIST_CF_FINAL_REPORT.json` is the terminal record.
