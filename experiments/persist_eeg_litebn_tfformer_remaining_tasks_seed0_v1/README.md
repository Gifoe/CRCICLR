# Original TFFormer remaining seed0 tasks

This experiment keeps the committed TFFormer architecture unchanged and runs seed 0 on OpenBMI ERP, OpenBMI MI, and WBCIC MI. Checkpoints use only canonical inner-validation subject-equal balanced accuracy. Each existing exposed/internal-heldout diagnostic is evaluated only after all five checkpoints for its task are frozen.

Because the original TFFormer SSVEP result did not pass 0.9234, these later-task results are explicitly additional diagnostics and are not represented as a valid pass through the earlier sequential gate.
