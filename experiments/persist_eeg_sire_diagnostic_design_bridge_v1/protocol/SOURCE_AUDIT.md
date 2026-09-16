# Source audit

100 frozen checkpoint cells verified by SHA256: 40 Part A and 60 Part B.
Part A reuses the exact seed0 Full/B1 checkpoint, normalizer, embedding, Protected union and PSWA random-control rule.
Part B uses server's historical Full as B0 at the user's explicit direction; split and train-only normalizer match but training protocol is NOT matched to B1/B2/B4.
No neural retraining is planned. The Part B comparison is descriptive and cannot isolate architecture effects.
The 40 outer-development subjects form a non-overlapping five-fold partition (8 per fold).
Part A evaluates previously exposed 14-person internal-heldout or WBCIC 10-person true-outer cohorts.
Part B must not read outer-development outcomes until SELECTION_FREEZE.json exists.
Checkpoint and normalizer paths/hashes, splits and exact source details are in SOURCE_MANIFEST.json.
