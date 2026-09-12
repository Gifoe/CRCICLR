# Exact LiteBN current-runtime replay control

This is an attribution control, not a model proposal. It replays exact recovered
historical LiteBN seed0 initialization, original OpenBMI-MI manifests and
fold-specific normalizers in the current Windows/Torch/CUDA runtime. The only
intended changed factor is execution runtime. Five OpenBMI-MI folds run before
any interpretation; no other dataset or seed is in scope.

The terminal label is produced only after all five folds, outer evaluation,
the already-open internal heldout diagnostic, and read-only same-runtime
candidate recalibration.
