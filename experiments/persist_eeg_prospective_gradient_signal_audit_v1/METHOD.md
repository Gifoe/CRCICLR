# Prospective Gradient Signal Audit v1

This is a frozen diagnostic audit, not a new method. It reuses the PMG-fast seed-0 canonical EEGNet M0 model-fit-only anchors and the recorded five pseudo-environment meta-fold partitions. For each dataset, outer fold, meta-fold, and ten deterministic paired subject-balanced draws, it measures source-to-pseudo-future gradients at the frozen parameters and actual relative-displacement responses at epsilon 1e-5, 1e-4, and 1e-3. No optimizer step is present in the primary path.

The optional WBCIC fold-1 forensic reproduction is separate and is executed only because the PMG-fast runtime contains no M2 checkpoint or step log. It uses the original locked PMG-fast recipe once, without repair or parameter changes, to localize the known collapse. If post-processing is interrupted after the primary table is persisted, --resume-primary continues from those immutable rows without recomputing them.
