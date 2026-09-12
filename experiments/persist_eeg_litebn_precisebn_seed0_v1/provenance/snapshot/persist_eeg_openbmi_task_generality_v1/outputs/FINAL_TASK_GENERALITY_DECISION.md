# PERSIST-EEG OpenBMI Task-Generality Test

The final fusion rule was frozen from the MI experiments before ERP and SSVEP outcomes were evaluated.

## 1. Protocol

- Same OpenBMI 40 SEARCH / 14 held-out membership; cache-session 1 to cache-session 2.
- EEGNet and LiteBN trained from scratch over 5 folds × 3 seeds.
- Fixed 50/50 logit fusion; no task-specific fusion tuning.

## 2. Main held-out table

| Task | Classes | EEGNet BA | LiteBN BA | LOGIT50 BA | Δ vs EEGNet | Median Δ | 95% CI | Positive subjects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MI | 2 | 0.7229 | 0.7490 | 0.7557 | +3.281 pp | +2.833 pp | [+1.519, +5.029] pp | 11/14 |
| ERP | 2 | 0.8482 | 0.8480 | 0.8570 | +0.886 pp | +0.747 pp | [+0.599, +1.214] pp | 14/14 |
| SSVEP | 4 | 0.8908 | 0.9102 | 0.9195 | +2.876 pp | +1.533 pp | [+1.381, +4.581] pp | 10/14 |

## 3. Prospective hypothesis

H_TASK: LOGIT50 > EEGNet on both ERP and SSVEP held-out mean BA.

Answer: **SUPPORTED**.

## ERP

Mean gain: +0.886 pp; median: +0.747 pp; 95% CI [+0.599, +1.214] pp; positive subjects: 14/14.
Harmful tail (≤−1/−3/−5 pp): 0/0/0; complementarity fraction: 0.0952.
Classification: `POSITIVE_TASK_GENERALIZATION`.

## SSVEP

Mean gain: +2.876 pp; median: +1.533 pp; 95% CI [+1.381, +4.581] pp; positive subjects: 10/14.
Harmful tail (≤−1/−3/−5 pp): 0/0/0; complementarity fraction: 0.0874.
Classification: `STRONG_TASK_GENERALIZATION`.

## 6. Is the frozen fusion benefit MI-specific?

**NO** under the locked ERP/SSVEP task test.

## 7. Does carrier complementarity appear in all three OpenBMI tasks?

**YES**.

## 8. Overall terminal

`OPENBMI_TASK_GENERALITY_POSITIVE`

## 9. Scientific interpretation

The primary unit is the held-out subject after averaging 15 fixed carrier-pair replicates. ERP and SSVEP were run under the single protocol lock after both task grids completed. The fixed predictor has no learned fusion parameter and no target-session adaptation. The terminal follows the preregistered subject-level criteria without exclusions or post-outcome changes. These results assess constructive task generality and carrier complementarity, not universality of every historical PERSIST mechanism.
