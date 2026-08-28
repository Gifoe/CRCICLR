# Iteration ledger

## Pre-outcome definition repair

The prompt allowed cosine similarity between class-conditioned logit-shift vectors. For binary logits, centering produces a rank-one vector `[a, -a]`; cosine collapses largely to a sign test and is mathematically degenerate as a graded stability measure.

Before any Stage-0.5 S3 association was computed, the definition was repaired to a class-conditioned standardized mean shift of the correct-class margin. The same non-degenerate construction is used separately for anchor decision stability and adaptation-effect stability. The prediction was fixed in advance: smaller standardized S1-validation-to-S2 shift (a score closer to zero) should imply more reliable future utility.

No feature, fold, outcome, model, threshold rule, or adapter was changed after S3 association became visible in this experiment.

