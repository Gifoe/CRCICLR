# Fold-role audit

For fold k, outcome=F_k, validation/discovery=F_(k+1 mod 5), and model-fit is the remaining three frozen folds. Training uses model-fit S1+S2; early stopping/lambda selection uses validation/discovery S3; outcome uses outcome S3 only after source freeze. Every authorized subject is outcome exactly once and validation/discovery exactly once; all roles are subject-disjoint.
