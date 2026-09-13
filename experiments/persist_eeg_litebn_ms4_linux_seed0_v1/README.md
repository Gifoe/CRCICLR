# LiteBN-MS4, WBCIC-MI seed 0

This experiment extends the validated C0 DualCE training path from two to four
stochastic supervised forwards. It changes training compute only: the model,
parameters, buffers, checkpoint selection, and one-pass inference graph remain
the exact historical LiteBN.

Scope is locked to WBCIC-MI, seed 0, five canonical folds. B0 and C0 are reused
from the authoritative controlled-v2 experiment. The already-open internal
heldout cohort is development/model-selection data, not a sealed final test.

Run with `code/run_ms4_wbcic.py`. The run is resumable at epoch boundaries and
freezes all five selected checkpoints before outer or internal-heldout access.
