# Early-stop versus fixed-60 descriptive audit

This is descriptive only; the fixed 60-epoch continuation was prescribed before any new outer evaluation.

| Model | Fold | Previous selected | Full60 selected | Previous inner S2 BA | Full60 inner S2 BA | Changed |
|---|---:|---:|---:|---:|---:|---|
| EEGNet | 0 | 14 | 50 | 0.923529 | 0.933091 | yes |
| SIRE-EEG | 0 | 31 | 31 | 0.928200 | 0.928200 | no |
| EEGNet | 1 | 25 | 25 | 0.940754 | 0.940754 | no |
| SIRE-EEG | 1 | 22 | 22 | 0.949593 | 0.949593 | no |
| EEGNet | 2 | 31 | 31 | 0.919300 | 0.919300 | no |
| SIRE-EEG | 2 | 19 | 43 | 0.917617 | 0.924063 | yes |
| EEGNet | 3 | 24 | 39 | 0.911570 | 0.912171 | yes |
| SIRE-EEG | 3 | 24 | 60 | 0.902048 | 0.906279 | yes |
| EEGNet | 4 | 10 | 59 | 0.938043 | 0.948596 | yes |
| SIRE-EEG | 4 | 26 | 54 | 0.941034 | 0.944771 | yes |
