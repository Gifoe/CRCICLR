# GeoSR frozen protocol

## Scope

- OpenBMI MI and the 41-subject WBCIC development cohort only.
- Canonical folds 0--4; biological subject is the inference unit.
- Phase A starts with scientific seed 0 and canonical EEGNet only.
- WBCIC outer-10 and OpenBMI sealed/outer confirmation data remain closed.

## Model and optimization

VanillaEEGNet is unchanged (F1=8, depth multiplier=2, F2=16, temporal
kernel=64, depthwise kernel=16, average pools 4 and 8, dropout=.25,
64-dimensional ELU/LayerNorm embedding, linear 64-to-2 head).  AdamW uses
learning rate 3e-4, weight decay 5e-4, batch size 64, gradient clipping 5,
maximum 60 epochs, minimum 10 epochs, and patience 8.  Epoch selection is
discovery mean biological-subject BA, then lower NLL, then earlier epoch.

## Cross-fitted source risk

Within each outer fold, the model-fit source subjects are partitioned into
five deterministic SHA-256 folds.  Each held-out subject has a teacher trained
without that subject; teacher normalizers and epoch selection use only the
teacher training subjects.  In that teacher coordinate system:

`N_geo = 1 - .5*(cos(v_s,t2,c_t1) + cos(v_s,t1,c_t2))`.

`N_loss` is balanced class NLL on held-out descriptor trials.  Subject ranks
use `scipy.stats.rankdata(method="average")`.  GeoSR weight is
`0.5 + 0.5*r_geo + 0.5*r_loss`; the nominal subject-weight range is [0.5,1.5].
Descriptor support is metadata-frozen at 32 trials per class/session.

## Methods and gates

Pre-registered methods are CANONICAL_ERM, SUBJECT_BALANCED_ERM, RANDOM_RANK,
LOSS_HARD, GEO_ONLY, and GEOSR.  All students use the same initial state and
minibatch order schedule.  The final refit recomputes source cross-fit risk on
model-fit plus discovery subjects and trains exactly the selected epoch count.

Seed-0 GO requires all six gates: mean utility, fixed bottom-25% utility,
individual-harm limits, fold consistency, no worse than RANDOM_RANK, and
paired subject-bootstrap upper CI > 0 on both datasets.  Only a complete GO
permits automatic seeds 1 and 2; otherwise the experiment stops.
