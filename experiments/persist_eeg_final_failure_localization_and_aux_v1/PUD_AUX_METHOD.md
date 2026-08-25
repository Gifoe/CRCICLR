# PUD-Aux method

Single-path EEGNet F8/F16, embedding 64, dropout 0.25, with a training-only linear auxiliary head predicting centered normalized frozen teacher targets. Lambda selected on source inner validation from {0.05, 0.10, 0.25}; controls Random-Aux, Identity-Aux, Full-Teacher-KD-Aux and P-only-Aux. No holdout or WBCIC outer access.
