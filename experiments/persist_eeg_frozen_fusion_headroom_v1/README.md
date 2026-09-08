# Frozen EEGNet + LiteBN fusion/headroom audit

This is an inference-only audit of the completed five-fold, three-seed carrier
experiment. It validates frozen checkpoint provenance, reproduces the recorded
EEGNet/LiteBN fold-seed balanced accuracies, then applies only pre-declared
50/50 logit and probability fusion. No optimizer, backward pass, fitting,
calibration, gate, or fusion-weight search is permitted.
