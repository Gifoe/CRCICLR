# BNCI2015-001 seed-0 result summary

Primary estimates are biological-subject-equal means across the 12 outer-test subjects. The paired bootstrap has 20,000 biological-subject resamples.

| Model | Future S2 BA | Future S2 Macro-F1 | WS-BA |
|---|---:|---:|---:|
| EEGNet | 0.6350 | 0.5909 | 0.6233 |
| SIRE-EEG | 0.6642 | 0.6395 | 0.6333 |

## Paired SIRE-EEG minus EEGNet

| Metric | Mean difference | 95% bootstrap CI | Unit | N |
|---|---:|---:|---|---:|
| future S2 BA | +0.0292 | [+0.0033, +0.0575] | biological subject | 12 |
| WS-BA | +0.0100 | [-0.0188, +0.0408] | biological subject | 12 |

## Selected checkpoints and outer S2 BA

| Fold | Model | Selected epoch | Inner-val S2 BA | Outer S2 BA | Checkpoint SHA-256 |
|---:|---|---:|---:|---:|---|
| 0 | EEGNet | 58 | 0.8700 | 0.6383 | `10d313fd65a39f38db1a148b7ce7793192a778292f3ef4026e3c6086b238d64f` |
| 0 | SIRE-EEG | 12 | 0.8500 | 0.6333 | `3171ab8927c0a5d9f9e829dc139b2a9969e87a5b6794af0a1a704f10979d2e1a` |
| 1 | EEGNet | 10 | 0.5300 | 0.5533 | `a5aa49416b6db3c99f205bb9a0a1b3b05bd466b8f7f17066e9ad12bbb21b8ab5` |
| 1 | SIRE-EEG | 52 | 0.5600 | 0.6333 | `ba653ad48822c5bd538525816de52a16cc18861e9cc56aca38ea343a0bd5d767` |
| 2 | EEGNet | 46 | 0.6450 | 0.8200 | `d2a19fb6287b0275c7a7a3b2cef15737a75699aa17076958a053587c87e559b3` |
| 2 | SIRE-EEG | 11 | 0.6900 | 0.8175 | `88c609ef3b1f7edf22903cdfbbf93048428c1d003b6ef63dfda3fb6878061c3f` |
| 3 | EEGNet | 53 | 0.8550 | 0.5175 | `6c7e788bf233aee9562bb433e7e99392b6bfe054e53c09e9716ab359916096ce` |
| 3 | SIRE-EEG | 10 | 0.8750 | 0.5100 | `3cbb46e820f8fc9e63688e430c6af718a3f4829a43164e476db35a5e86b03a87` |
| 4 | EEGNet | 21 | 0.6650 | 0.6850 | `8fac1376e380ed4fe9bf74366bd5cf6200b7b7d475fe59cf5c2e870ef87082c6` |
| 4 | SIRE-EEG | 17 | 0.6800 | 0.7575 | `cf36ebcc84618627a4e93bcc247b89987a1bebe7441837ede658f2956e8c1b6c` |

Hard stop honored: this report contains seed 0 only; no diagnostics, alternative preprocessing, other baselines, or additional seeds were run.
