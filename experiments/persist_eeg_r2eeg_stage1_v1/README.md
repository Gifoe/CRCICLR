# R2EEG Stage-1 seed-0

This experiment is a fixed-split, SEARCH-only prospective test of a new encoder
trained from the existing signal-level epoch caches. It compares EEGNet-ERM,
R2EEG-ERM, and R2EEG-Rel for exactly three frozen subject-disjoint folds on
OpenBMI and WBCIC. Runtime checkpoints are deliberately outside the repository.

Run with `python code/run_stage1.py --device auto`; use `--validate-only` before
any training. The runner never enumerates or loads V8 internal holdout or WBCIC
true-outer subjects.
