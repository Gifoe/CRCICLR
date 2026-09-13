# LiteBN-XC Linux seed-0 experiment

This experiment tests one fixed candidate: the frozen LiteBN-XNG architecture
plus only the authoritative LiteBN-XS shared per-electrode channel gate. The
temporal-scale gate remains absent.

Phase 1 runs WBCIC-MI seed 0 over the five canonical folds, freezes all XC
checkpoints, evaluates outer development against the existing frozen Linux XNG
reference, and then evaluates both methods on the already-open internal
heldout development diagnostic. No new sealed final cohort is accessed.

Runtime checkpoints live outside Git at
`/root/rivermind-data/litebn_xc_linux_seed0_runtime`.
