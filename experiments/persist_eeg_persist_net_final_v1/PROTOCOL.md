# Frozen experimental protocol

## Primary design

OpenBMI MI V8_SEARCH (40 subjects), deterministic five subject-disjoint folds,
three seeds.  Each outer fold contains eight outcome subjects.  The remaining
32 are source subjects; eight are a deterministic inner validation set and 24
are inner model-fit subjects.  Source training can use Sessions 1 and 2.
Target adaptation uses only the outcome subject's Session 1 labels.  Session 2
is evaluation-only.

## Source teacher and baseline selection

Two EEGNet widths are compared on inner source validation only.  Early stopping
uses mean subject BA on inner-validation Session 2.  The selected architecture
and epoch are refit on all 32 source subjects.  No historical checkpoint is
loaded.  Target adaptation is selected from the predeclared V6-derived set in
`PROTOCOL_FROZEN.json`, using inner-source S1-to-S2 episodes only.

## Certification

Teacher embeddings are source-whitened.  Class-conditioned subject/session
centroids define a symmetric cross-session persistence spectrum.  Individual
directions must exceed a source-only session-permutation null and P>=0.05.
Signed utility is the subject-level teacher CE increase under finite erasure
minus 64 matched-energy random erasures; its paired subject-bootstrap lower
95% bound must exceed zero.  Exact finite decision dependence is the Exp3
class-centered logit RMS and must exceed the matched random mean.  No identity
quantity enters the PUD definition.  Rank is the number passing these fixed
filters; there is no top-8 or post-hoc cap.

## Model and optimization

The student has independent protected/adaptive EEGNet branches.  The branch
width is chosen mechanically as the parameter-nearest of two predeclared
splits while remaining <=1.25x B1.  It is not chosen by outcome accuracy.
Default loss weights are lambda_D=1, lambda_R=1, lambda_P=0.1.  No scientific
grid is executed unless V0 detects one of the predeclared engineering
pathologies.  A single V1.1 repair is allowed only for collapse, non-finite
scale, missing gradient, or freeze violation.

## Ablations and robustness

B0/B1/B2, dual control, PUD all-adapt, identity, random, and FULL run for all
five folds and three seeds.  P-only, P+U, P+D, and PCA run for all five folds
at seed 0 as secondary ablations.  This asymmetry is frozen before outcome
access and must be shown in the report.

## Statistics and gates

Subjects are the unit.  Primary paired differences use each subject's mean
across three seeds and 10,000 subject-bootstrap draws.  Fold and seed
consistency are reported separately.  G1-G9 are copied verbatim into the JSON
lock.  G1, G6, or G9 failure stops before WBCIC.  No OpenBMI internal holdout is
opened under any result.
