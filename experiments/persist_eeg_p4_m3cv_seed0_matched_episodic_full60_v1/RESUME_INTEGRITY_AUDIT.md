# Resume integrity audit

Every source checkpoint was copied byte-for-byte to the separate full-60 runtime before resumption. The existing normalizer files were loaded, not refit. `normalizer_semantic_sha256` is the digest stored in the checkpoint; `normalizer_file_sha256` is the physical NPZ-file digest.

| Model | Fold | Previous latest | Resume start | Final | Previous best | Full60 selected | Changed |
|---|---:|---:|---:|---:|---:|---:|---|
| EEGNet | 0 | 22 | 23 | 60 | 14 | 50 | yes |
| SIRE-EEG | 0 | 39 | 40 | 60 | 31 | 31 | no |
| EEGNet | 1 | 33 | 34 | 60 | 25 | 25 | no |
| SIRE-EEG | 1 | 30 | 31 | 60 | 22 | 22 | no |
| EEGNet | 2 | 39 | 40 | 60 | 31 | 31 | no |
| SIRE-EEG | 2 | 27 | 28 | 60 | 19 | 43 | yes |
| EEGNet | 3 | 32 | 33 | 60 | 24 | 39 | yes |
| SIRE-EEG | 3 | 32 | 33 | 60 | 24 | 60 | yes |
| EEGNet | 4 | 18 | 19 | 60 | 10 | 59 | yes |
| SIRE-EEG | 4 | 34 | 35 | 60 | 26 | 54 | yes |
