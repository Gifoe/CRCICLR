# Scientific rationale

Earlier experiments showed that subject identity is not a sufficient proxy for
task-consequential variation, and that erasing or transporting representation
structure did not reliably improve future-session balanced accuracy.  PERSIST-RE
therefore quarantines subject-specific *decision deviations* in a centered
random-effect branch while directly training the shared population predictor on
pseudo-unseen subjects.  The method is only supported if the population-only
predictor beats matched generic controls under a paired biological-subject
bootstrap.

