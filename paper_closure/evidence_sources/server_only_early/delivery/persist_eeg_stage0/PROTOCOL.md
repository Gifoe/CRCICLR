# PERSIST-EEG Stage-0 frozen protocol

Frozen before inspecting scientific results. This study is a falsification study; negative
results are retained and thresholds are not changed after freeze.

## Scientific questions

1. Does subject-persistent EEG information survive session and paradigm changes?
2. Conditional on subject, does session/recording-persistent information remain?
3. Do generic representations jointly contain long-, recording-, and event-varying information?
4. Does removal of a training-subject-derived persistent subspace improve unseen-subject event decoding?

## Primary data and representations

- OpenBMI / Lee2019, subjects 1–54, sessions 1–2, MI/ERP/SSVEP from the same cohort.
- Handcrafted spectral, temporal, shrinkage covariance/correlation/tangent features.
- A shared multi-task EEGNet and a shared multi-task two-block ConvTransformer.
- Five subject-disjoint outer folds. Neural seeds: 0, 1, 2, 3, 4.

No test subject contributes to normalization, early stopping, model selection, tangent
reference, subject subspace, or erasure-rank selection. Original OpenBMI train/test phases
are recording phases, not ML splits.

The raw-condition event probe selects regularization on validation subjects once per task;
all matched erasure conditions reuse that C to prevent control-specific retuning. MOABB 1.5's
Lee2019 generic session filter is bypassed because its selected keys (1,2) do not match its
raw keys (0,1); the underlying official subject loader is used before that faulty filter.
MOABB's Windows sanitizer is patched locally to preserve the absolute drive anchor; without
this, it rewrites `E:` to a relative `E-` directory.

For neural training, one balanced epoch is a full pass through the shortest paradigm. Every
update contains one MI, ERP, and SSVEP batch with equal loss weight. Longer-paradigm indices
are traversed in deterministic cyclic shards, and early stopping is disabled until every
training trial has appeared at least once. Thus ERP cannot set the update count or dominate,
but no trial, subject, fold, seed, or paradigm is dropped.

## Primary metrics and fixed gates

- Long persistence: cross-session/cross-paradigm same-subject verification AUROC (Gate A).
- Medium persistence: within-subject session verification AUROC (Gate B).
- Event utility: unseen-subject balanced accuracy and paired delta after erasure (Gate C).
- Conditional external replication: EEGMMIDB execution/imagery verification and MI erasure delta (Gate D).

Exact numeric criteria are serialized in `METHOD_FREEZE.json` and are not inferred from
results. Subject is the resampling unit: 2,000 bootstrap samples and 1,000 subject-block
permutations. Random-subspace controls use 20 repetitions.

## Stopping rule

All preregistered OpenBMI representation experiments run before the OpenBMI decision.
EEGMMIDB opens only if OpenBMI Gates A, B, and C support continuation. Any failed core gate
produces an explicit NO-GO label rather than an inconclusive result or a method rescue.

## Interpretation boundary

The operational terms are long-persistent, recording/session-persistent, and event-varying
information. They are not claims of personality, pure physiological traits, electrode
impedance, or latent cognition.
