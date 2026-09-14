# LiteBN-CM-GA final-candidate experiment

This branch tests exactly one candidate: the historical 64-feature LiteBN with
an identity-initialized channel residual (C), three identity-initialized
temporal residual mixers (M), and cross-subject Gradient-Admissible training.

Stage 0 must first establish that cross-subject gradient conflict in the exact
C/M parameter space predicts prospective harm for both OpenBMI MI and WBCIC
MI. Failure on either dataset terminates the experiment before candidate
training.

The V8 internal and WBCIC true-outer resources are treated only as previously
exposed benchmark evaluation. They are not opened during Stage 0. No new
sealed cohort is accessed.

