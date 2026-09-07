# R2EEG SpatialRepair, OpenBMI fold 0

This is a one-fold, seed-0, ERM-only carrier diagnostic. It reuses the frozen
R2EEG-v1 OpenBMI SEARCH split, cache, normalizer and 128-trial episode manifest.
It never loads reserved V8 holdout or WBCIC outer data. It stops after OpenBMI
fold 0 and is not evidence of cross-dataset generalization.
