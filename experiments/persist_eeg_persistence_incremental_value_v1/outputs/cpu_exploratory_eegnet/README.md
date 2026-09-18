# CPU exploratory PU versus U-only analysis

## Status

`CPU_EXPLORATORY_NON_EQUIVALENT` — this package is a direction-diagnostic
analysis of the locked EEGNet checkpoints and existing Experiment 1 selector
coordinates. No neural network was trained, no checkpoint was modified, and no
heldout outcome was used in selection. It must **not** be reported as a
numerically equivalent continuation of the original CUDA execution.

The local CPU shadow test reproduced checkpoint SHA, normalizer SHA, active
rank, representation dimension, and selector coordinates, but changed some
per-subject effects by up to 3.141 pp. That discrepancy is why the present
files are isolated from formal result directories.

## Completed scope

EEGNet: 4 tasks x 5 folds x 3 seeds = 60/60 explanatory cells.

EEGConformer: 4 tasks x 5 folds x 3 seeds = 60/60 explanatory cells. Two
WBCIC-MI cells are valid `EMPTY_PU` controls, so there are 13 estimable cells
for that model/task rather than 15.

Raw EEG, caches, checkpoints, and runtime cell JSON are excluded.

## Directional findings

All point estimates below aggregate repeated fold/seed observations within each
biological heldout subject before the 20,000-draw subject bootstrap.

| Model | Criterion | OpenBMI-MI | OpenBMI-ERP | OpenBMI-SSVEP | WBCIC-MI |
|---|---|---:|---:|---:|---:|
| EEGNet | Persistent rank fraction within U-only | 100% | 100% | 83.3% | 100% |
| EEGConformer | Persistent rank fraction within U-only | 100% | 93.3% | 100% | 100% |
| EEGNet | U-only minus PU future BA | 0.00 pp | 0.00 pp | +1.02 pp | 0.00 pp |
| EEGConformer | U-only minus PU future BA | 0.00 pp | -0.03 pp | 0.00 pp | 0.00 pp |
| EEGNet | Nonpersistent conditional excess future contribution | not estimable | not estimable | +6.23 pp | not estimable |
| EEGConformer | Nonpersistent conditional excess future contribution | not estimable | -1.18 pp | not estimable | not estimable |
| EEGNet | Utility-matched persistent minus nonpersistent future retention | not estimable | +3.76 pp | -2.08 pp | not estimable |
| EEGConformer | Utility-matched persistent minus nonpersistent future retention | not estimable | +1.99 pp | not estimable | not estimable |

The desired universal success pattern is not supported even in this exploratory
diagnostic. U-only is mostly persistent and is commonly identical to PU, but
the EEGNet SSVEP nonpersistent component is not negligible (+6.23 pp; 95% CI
[+5.00, +7.47]), while the utility-matched future-retention contrasts are not
consistent: EEGNet ERP is positive (+3.76 pp; [+0.22, +7.22]), EEGNet SSVEP is
negative (-2.08 pp; [-4.67, +0.46]), and EEGConformer ERP is inconclusive
(+1.99 pp; [-0.62, +4.66]).

See `UTILITY_MATCHED_TASK_SUMMARY.csv` and the subject-level compact CSV files
for confidence intervals and all aggregated inputs.
