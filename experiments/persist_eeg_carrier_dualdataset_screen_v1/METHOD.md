# Carrier dual-dataset screen

Seed-0, frozen Stage-1 development fold-0 architecture screen. Canonical EEGNet, the exact historical Compact encoder, Compact-Lite-BN, Compact-Lite-GN, and two frozen-EEGNet multi-scale raw-signal residual branches are trained from the same signal-level cache, normalization, episode manifest, optimizer, and 60-epoch budget. The residual RMS variant changes only its explicit per-trial scalar RMS normalization.

All architectures and decision gates are fixed before outer-development evaluation. The experiment accesses only V8_SEARCH development subjects; V8 internal holdout and WBCIC true outer data are excluded.
