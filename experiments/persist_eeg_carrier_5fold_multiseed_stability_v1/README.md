# LiteBN 5-fold x 3-seed carrier stability

This experiment compares only the previously frozen canonical EEGNet and CompactLite-BN (LiteBN) carriers on V8_SEARCH OpenBMI and WBCIC data. It uses a fresh deterministic five-fold subject split, three seeds, future-session inner validation for checkpoint selection, and subject-level outer-development evaluation.

The historical three-fold seed-0 observations are context only and are not pooled with this estimate. No new architecture or mechanism is introduced.
