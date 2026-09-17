# BNCI2015-001 Full-model matched protocol: three seeds

Seed 0 is frozen reuse. Seeds 1/2 were trained with the authoritative final convention: fixed seed-0 folds and manifests, initialization RNG = seed, training RNG = 100000 + seed.

| Seed | Model | Future S2 BA | Future S2 Macro-F1 | WS-BA |
|---:|---|---:|---:|---:|
| 0 | EEGNet | 0.6350 | 0.5909 | 0.6233 |
| 1 | EEGNet | 0.6471 | 0.6065 | 0.6142 |
| 2 | EEGNet | 0.6562 | 0.6363 | 0.6267 |
| 0 | SIRE-EEG | 0.6642 | 0.6395 | 0.6333 |
| 1 | SIRE-EEG | 0.6196 | 0.5829 | 0.5912 |
| 2 | SIRE-EEG | 0.6250 | 0.5792 | 0.6054 |

Three-seed SIRE−EEGNet paired effects are in `multiseed_summary.csv`; the bootstrap unit is the biological subject after within-subject seed averaging.
