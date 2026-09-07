# HSC-TTA CPU Critical-Index Repair Report

## Outcome

CPU GO: **PASS**. GPU work was intentionally not started.

- Tests: 60 passed.
- Overall line coverage: 78% (previous baseline 71%).
- Formal certificate module: 97%.
- Critical-index predictor module: 90%.
- U-only selection module: 90%.
- Critical-index simulation module: 96%.
- Artifact validation: `pass`, 0 failures.

## Formal changes

The main target is now empirical prediction-set miscoverage on the immutable future episode `V_s-main`. The formal predictor estimates the alpha-specific critical lambda index. Calibration uses one residual per subject, obtained by maximizing across the three actions only. The old empirical-Bernstein upper-risk code remains explicitly marked as supplementary diagnostic and is absent from formal training, scoring, feasibility, and selection APIs.

The lambda grid has 20 nontrivial values plus a `lambda=1.0` full-set sentinel. The sentinel has risk zero, is reported as uncertified fallback, and is excluded from nontrivial CSR. Selection requires `n_classes`, rejects future outcome columns, and uses context utility plus fixed adaptation cost only.

## Real CPU artifacts

- HMC: 151 subjects × 5 seeds; context 180 epochs; V-main exactly 240 epochs; zero exclusions.
- CAP: 103 subjects × 5 seeds; context 175–180 epochs; V-main exactly 240 epochs; zero exclusions.
- EEGMMIDB: 109 subjects × 5 seeds; official runs 4/6 versus 8/10/12/14; zero exclusions.
- Fifteen immutable files were written under `data/episodes_main120`; the original `data/episodes` files were not overwritten.
- Fifteen deterministic internal split JSON files were written under `data/splits_internal`.
- HMC fit/val is 60/10 and meta-risk folds cover 35 subjects; MI fit/val is 38/7 and folds cover 30 subjects.
- CAP contains no task-head or meta-predictor training subset and explicitly inherits HMC.

## Frozen HMC→CAP channel protocol

The availability-only rule selected **C4** with hash `2e35eff22ad71af3cf30612602934a97f5b0cb610ce60fa25fed87f7b5bc71eb`. HMC retains 151 subjects and CAP retains 99 subjects. Four CAP subjects lack the selected C4 derivation; this limitation is recorded rather than hidden or repaired by channel duplication.

## Simulations A–G (500 Monte Carlo repetitions)

| alpha | selected coverage ± MCSE | nontrivial CSR ± MCSE | q saturation rate |
|---:|---:|---:|---:|
| 0.10 | 0.967700 ± 0.001403 | 0.992275 ± 0.000546 | 0.000000 |
| 0.20 | 0.969200 ± 0.001422 | 0.999875 ± 0.000056 | 0.000000 |

Under adversarial U-only action selection, pointwise action calibration covered 0.753275 ± 0.004281, while actionwise simultaneous calibration covered 0.905375 ± 0.003204. Predictor-bias experiments at -2, 0, and +2 critical-index units all remained above nominal selected coverage after conformal correction.

## CPU GO checks

1. Feasible alpha=0.20 nontrivial CSR is positive: PASS.
2. Selected-risk coverage is not below nominal minus Monte Carlo tolerance: PASS.
3. Critical-index q does not saturate at the full-set index: PASS.
4. All tests pass: PASS.
5. Real main120, split, and channel artifacts reproduce with zero validation failures: PASS.

These are synthetic validity checks and engineering evidence, not real EEG performance claims. No final-test labels, GPU embeddings, task-head results, or TTA outcomes were generated during this CPU phase.
