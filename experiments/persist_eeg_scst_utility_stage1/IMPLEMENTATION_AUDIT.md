# Implementation audit

The Stage-0 `ATCNet` is renamed `ATCNet-CleanRoom`. It contains one shared
multi-head attention block, three hand-selected temporal windows, a shared
two-block TCN, and one shared classifier. Braindecode ATCNet uses five windows,
separate attention/TCN/classifier modules per window, max-norm classifiers, and
the published convolution/pooling layout. These are material differences, so
the old result is preserved and `ATCNet-Official` is evaluated separately.

`EEGNeX` uses Braindecode 1.2.0 with its documented defaults. No approximate
replacement is introduced. Both official models use the frozen Stage-0
per-trial, per-channel temporal standardization and source-only fold roles.

