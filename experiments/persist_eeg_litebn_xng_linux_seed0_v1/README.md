# LiteBN-XNG Linux seed-0 experiment

This experiment evaluates one fixed candidate: the authoritative LiteBN-X with
only the sample-dependent temporal-scale gate removed. The first execution
stage trains WBCIC-MI seed 0 over the five canonical folds. The exact LiteBN
matched-runtime baseline is deliberately deferred and must be trained before
any candidate-versus-baseline gate or internal-heldout comparison is reported.

The current `run_xng_task.py` entry point is candidate-only. It freezes all five
XNG checkpoints before opening outer-development subjects and does not access
the internal heldout cohort.

Runtime checkpoints are stored outside Git at
`/root/rivermind-data/litebn_xng_linux_seed0_runtime`.
