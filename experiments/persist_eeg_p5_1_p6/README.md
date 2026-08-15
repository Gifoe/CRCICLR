# PERSIST-EEG P5.1 + P6

This directory contains the protocol-completion run for PERSIST-ICG.

P5.1 performs the originally authorised TRAIN-only five-fold nested
subject-CV search for V0/V1/V2.  It never uses development-validation
subjects for hyperparameter or epoch selection.  If no selected version
passes the frozen PERSIST_ICG_VIABLE gates, P6 audits whether the frozen
Protected geometry is complementary as a readout score rather than as an
alignment objective.

Run from the repository root in the existing P1 GPU environment:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -u experiments/persist_eeg_p5_1_p6/code/p5_1_p6.py --phase all --device cuda
```

The runner reuses the verified P5 frozen-E0 feature caches, writes only to
`experiments/persist_eeg_p5_1_p6/outputs/`, and never reads outer-test
samples or labels.

## Completed result

The completed 3-fold x 2-seed run terminates at:

`PERSISTENCE_GEOMETRY_HAS_NO_DECISION_FUSION_HEADROOM`

P5.1 executed all 216 candidate summaries and 1,080 subject-disjoint inner
fold fits. TRAIN-only tuning did not make any ICG version viable:

| Version | Mean paired Delta BA | Positive runs | Hierarchical 95% CI |
|---|---:|---:|---:|
| V0 | -0.000926 | 2/6 | [-0.002963, 0.000833] |
| V1 | -0.000648 | 3/6 | [-0.002593, 0.000926] |
| V2 | -0.001852 | 0/6 | [-0.004352, -0.000093] |

The original rule did not authorize V3. P6 therefore used the V2 matched
control selected by TRAIN-inner control BA. Frozen Protected-score fusion
had mean Delta BA -0.001019 (0/6 positive; hierarchical 95% CI
[-0.003611, 0.000000]). Five runs selected alpha=0; the only run selecting
alpha=0.25 lost 0.006111 BA. Complementary errors exist diagnostically, but
the predeclared deployable scalar fusion cannot exploit them.

The downloaded result package excludes only reconstructible geometry-target
caches. It includes every inner split, candidate/inner result table,
selected configuration, matched development-validation result, P6 random
draw/control table, checkpoint, final report, and a 22-check reproducibility
audit. Outer-test remains unused.

Run the local materialized-output audit with:

```powershell
python experiments/persist_eeg_p5_1_p6/code/audit_outputs.py
```
