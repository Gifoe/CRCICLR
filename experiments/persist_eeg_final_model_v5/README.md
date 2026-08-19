# PERSIST-EEG Final Model V5

V5 studies local expert competence under subject/session shift.  The primary
prospective benchmark is the authorized 41-subject WBCIC development scope;
OpenBMI is retained as an exploratory non-degradation benchmark.

Final development result: `CS-LGS` (Cross-Session Local-Geometry Stack) reached
WBCIC development BA 0.817782, or +1.099 percentage points over the frozen
`W1_RAW_LINEAR` current baseline.  The paired subject-bootstrap CI is
[+0.195,+2.028] pp, all five folds are positive, and five solver-seed reruns
produce identical predictions.  This is a development-only, adaptively
discovered result; the sealed outer cohort remains untouched.

The experiment starts from the frozen V4 evidence, reconstructs every current
baseline, audits disagreement/oracle structure, and then evaluates output-only,
EEG-context, cross-session-reliability, local-neighbourhood, ranking,
hierarchical, correlation-aware, and small attention selectors with nested
subject-disjoint validation.

Hard constraints:

- WBCIC sealed outer data are never enumerated or loaded.
- `OUTER_TEST_USED` is always `false`.
- Target S3 labels are evaluation-only.
- When target-subject history is used, it is limited to legal S1/S2 data and is
  reported explicitly.
- All gains are reported against the strongest current legal reference, not a
  historical EEGNet.

The selected model uses S1/S2-only subject-local heads in five frozen expert
representations, an S1/S2-only 8-30 Hz CSP context score, a fixed C=1 linear
stack, and an exact `W1_RAW_LINEAR` fallback on unanimous trials.  CSP is not
claimed as a competent standalone expert (its control is weak).  PERSIST is not
claimed to add predictive BA; it is retained only as the previously frozen
fail-closed ACTION safety veto.

Reproduce the final executed stages from `code/` with:

```powershell
python run_subject_adaptation.py
python run_reliability_stack.py
python run_refit_disagreement.py
python run_csp_augmentation.py --workers 4
python run_confirmation.py
python run_final_dev.py
```

Run commands and exact environment hashes are written to
`outputs/REPRODUCIBILITY.json` after the search is complete.
