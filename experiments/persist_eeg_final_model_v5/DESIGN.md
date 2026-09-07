# V5 research design

Primary target: at least +1.0 percentage point mean subject-balanced accuracy
over `B_STRONG_CURRENT` on WBCIC development, with a positive paired
subject-bootstrap lower bound, at least four positive grouped folds, majority
nonnegative subjects, and tolerable worst-subject behaviour.

The initial PRSE hypothesis is a B-strong-anchored competence model driven by
expert disagreement, train-only expert fingerprints, compact EEG context, and
cross-session reliability.  It is only a starting hypothesis.  Structurally
different alternatives are retained in the hypothesis ledger and failed
families are not deleted.

The initial zero-shot PRSE hypothesis failed: output-only, fixed EEG context,
kNN, multi-label correctness, pairwise ranking, minority rescue, and
correlation-aware aggregation did not transfer stably.  The successful redesign
uses legal target-subject history.  Each subject-local head is selected by
S1->S2 and S2->S1 validation only; held-out S3 labels and S3 batch statistics
are never used.  A weak CSP classifier contributes only a continuous context
score.  The final stack is fixed and leaves the frozen W1 decision unchanged
when the five raw experts are unanimous.

The protocol uses five subject-disjoint development folds.  Within every outer
fold, the next fold is calibration-only and the remaining subjects are
model-fit subjects.  Scaling, PCA, expert pruning, hyperparameter selection,
threshold selection, and reliability estimation are fitted without outer-fold
S3 outcomes.

For WBCIC future-session models, target-subject S1/S2 labels may be used as a
pre-S3 reliability signal because this is the frozen cross-session setting.
Target-subject S3 labels remain evaluation-only.  Results that use prior-session
labels are separated from zero-shot results.

The final development estimate is explicitly exploratory because successive
hypotheses were informed by development outcomes.  The specification is frozen
for a later explicitly authorized outer evaluation; V5 itself does not open,
enumerate, featurize, or score the outer subjects.
