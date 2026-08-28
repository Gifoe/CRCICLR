# SCST-DR protocol

## Scientific question

Can task decisions become robust to source-observed, label-preserving subject
variation without removing subject information?

## Hard gate

The first experiment is a transport-validity audit, not model training.  Four
competent historical MI settings are audited: OpenBMI/WBCIC crossed with
EEGNet/EEGConformer.  Five frozen subject-disjoint folds and historical ERM seed
0 are used.  The seed is not selected using transport outcomes.

Only two predeclared representation blocks are considered: the pooled backbone
state immediately before the learned embedding and the final task embedding.
The simpler final embedding is preferred whenever it passes every gate.

For each model-fit subject and class, Session 1 estimates a subject-class
residual.  The resulting direction is applied to independent Session-2 samples.
OpenBMI Session 2 is legal here only for model-fit subjects; outcome subjects and
the internal sealed 14 are not loaded.  WBCIC uses legal development-subject
Sessions 1/2; Session 3, outcome subjects, and the sealed outer 10 are not loaded.

The exact controls, estimators, quantitative gates, selection rule, and stop rule
are machine-locked in `protocol/STAGE0_PROTOCOL_LOCK.json` before the first
transport metric is computed.

