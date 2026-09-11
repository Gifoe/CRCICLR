# OpenBMI ERP LiteBN-XS seed stability (outer-development)

Only OpenBMI ERP was evaluated. Existing canonical folds were reused; no new inner split was created.
LiteBN baseline checkpoints were verified and reused. XS seed1/2 were trained independently with the original architecture and selection rule.
Outer-development rows were accessed for development/stability analysis. Final heldout was not accessed.

| Seed | LiteBN outer BA | XS outer BA | delta XS-LiteBN (pp) | LiteBN macro-F1 | XS macro-F1 | positive folds |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.832553 | 0.831735 | -0.082 | 0.784901 | 0.792438 | 2 |
| 1 | 0.831833 | 0.842386 | +1.055 | 0.784230 | 0.800768 | 3 |
| 2 | 0.834924 | 0.833977 | -0.095 | 0.792683 | 0.800516 | 3 |

Seed1/2 mean delta: +0.480 pp.
Positive seeds (0/1/2): 1/3.
Interpretation: ERP_DEGRADATION_MIXED.

Selection: max 60 epochs; canonical inner-validation subject-mean BA; eligibility starts epoch 10; patience 10; best checkpoint restored.

FINAL_HELDOUT_ACCESSED = NO
OUTER_DEVELOPMENT_ACCESSED = YES (development/stability analysis only)
