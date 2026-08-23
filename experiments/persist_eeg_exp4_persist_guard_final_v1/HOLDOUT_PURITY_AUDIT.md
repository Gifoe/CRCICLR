# Holdout purity audit

The historical 83.775% V6 checkpoint is **not legal** for the final 14-subject holdout: its training and selection predated the V8 40/14 partition. This experiment instead uses 15 repaired EEGNet checkpoints whose train/validation subject lists are subsets of V8_SEARCH and have zero holdout overlap. Normalization, protected-bank certification, risk fitting, threshold selection, and action selection are development/source-only. Internal holdout data and outcomes were not accessed.
