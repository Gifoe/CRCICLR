# V4 research design review

## What V3 established

V3 did not fail because the expert pool lacked headroom. B6 reached 0.846442
mean subject BA, while the KEEP-only oracle added 7.663 pp and the full global
action oracle added 8.596 pp. The combined menu added 10.702 pp. It failed
because prospective rescue/harm selectors converted rare switches into more
harm than rescue: the best M5 policy was -0.029 pp with a CI crossing zero.
Conditional ERASE correctness was discriminable (AUROC about 0.722), but the
candidate population was still harm-dominated. That makes selection and
aggregation—not action availability—the bottleneck.

## Modelling assumptions that were too weak

1. V3 predicted rescue and harm separately, then hard-switched. Separate
   probability errors are amplified by subtraction and thresholding.
2. It concentrated on action candidates before testing the stronger generic
   control: direct cross-fitted stacking of KEEP logits.
3. Hard selection throws away agreement information and makes one erroneous
   decision worth a full label flip. A bounded residual correction has a
   smaller failure surface.
4. Expert identity and variable expert count were reduced to aggregates. A
   permutation-invariant token model can estimate joint expert correctness
   without requiring the same expert count on both benchmarks.
5. PERSIST was used mainly as a flat feature vector. It may work better as a
   KEEP prior or an ERASE constraint than as ordinary trial-level signal.

## Evidence-informed candidate order

The first controls are subject-cross-fitted logistic stacking and shallow
boosting. Stacked generalization requires out-of-fold base predictions
(Wolpert, 1992, doi:10.1016/S0893-6080(05)80023-1; Ting & Witten, 1999,
doi:10.1613/jair.594). Dynamic ensemble selection literature treats local
competence estimation as the central problem rather than assuming the most
confident classifier is best (Cruz et al., 2018,
doi:10.1016/j.inffus.2017.09.010).

If direct stacking is insufficient, V4 tests B6-anchored residual logits,
joint expert correctness, and permutation-invariant expert aggregation.
Deep Sets (Zaheer et al., 2017, arXiv:1703.06114) and Set Transformer (Lee et
al., 2019, arXiv:1810.00825) motivate shared expert-token encoders, not a large
EEG encoder retrain. Learning-to-defer work motivates a conservative default
to B6 when the estimated utility gap is small (Mozannar & Sontag, 2020,
arXiv:2006.01862). SelectiveNet (Geifman & El-Yaniv, 2019,
arXiv:1901.09192) motivates reporting risk/switch coverage rather than only
accuracy. Calibration is evaluated explicitly because confidence ordering can
be useful while absolute probabilities are wrong (Guo et al., 2017,
PMLR 70:1321-1330).

## Falsifiable hypotheses

- H1: generic dynamic KEEP aggregation beats frozen B6 on grouped subjects.
- H2: action logits add gain beyond the best dynamic KEEP control.
- H3: PERSIST adds either BA or measurable safety beyond the matched
  KEEP+ACTION model.
- H4: bounded soft residual correction is safer than hard action switching.
- H5: the same information ladder transfers to WBCIC development subjects.

OpenBMI estimates remain exploratory. WBCIC outer is not read in V4.
