# LiteBN-SBTR seed0

| Method | Outer BA | Outer SD | Heldout BA | Heldout SD |
|---|---:|---:|---:|---:|
| Historical_LiteBN | 0.7915 | 0.0376 | 0.7449 | 0.0172 |
| SBTR_main_tail_aware | 0.6770 | 0.0449 | 0.6480 | 0.0212 |
| SBTR_meanBA_diagnostic | 0.6740 | 0.0468 | 0.6497 | 0.0235 |

Main SBTR outer ΔBA: -11.450 pp
Main SBTR heldout ΔBA: -9.686 pp
Positive outer folds: 0/5
Positive heldout folds: 0/5
Subject median ΔBA: -11.500 pp
Subject 25th percentile ΔBA: -16.250 pp
Subject 10th percentile ΔBA: -21.200 pp
Worst subject ΔBA: -25.000 pp

CURRENT_INTERNAL_HELDOUT_ACCESSED = YES
FINAL_HELDOUT_ACCESSED = NO

## Decision
STOP_SBTR

## Four required answers

1. Mean performance: no. Main SBTR is -11.450 pp on outer development and -9.686 pp on the current internal heldout cohort.
2. Fold variance: no. Outer fold SD rises from 0.0376 (LiteBN) to 0.0449 (SBTR); heldout fold SD rises from 0.0172 to 0.0212.
3. Lower-tail subjects: no. Outer subject deltas are -16.250 pp at the 25th percentile, -21.200 pp at the 10th percentile, and -25.000 pp worst-case.
4. Transfer: the outer degradation transfers to the current internal heldout diagnostic; there is no positive transfer.
