# M3CV seed-0 full-60 result summary

Dataset: M3CV / NEMAR `nm000166`.

- Downloaded subjects: 95; eligible subjects: 93; exclusions: `sub-035`, `sub-080`.
- Task: left- versus right-hand motor execution; S1=`ses-01`, S2=`ses-02`; input: 64 x 1000 at 250 Hz.
- All ten archived runs were resumed from their stored state and completed at epoch 60. Checkpoint selection uses maximum inner-val S2 subject-equal BA over epochs 10..60, with the earliest exact tie retained.

| Model | Future S2 BA | Future S2 Macro-F1 | WS-BA |
|---|---:|---:|---:|
| EEGNet | 0.9212 | 0.9198 | 0.8872 |
| SIRE-EEG | 0.9174 | 0.9165 | 0.8818 |

| Contrast | Delta | 95% CI | Improved / tied / harmed |
|---|---:|---:|---:|
| future S2 BA | -0.0037 | [-0.0095, +0.0021] | 36 / 9 / 48 |
| WS-BA | -0.0054 | [-0.0127, +0.0016] | 36 / 6 / 51 |

| Fold | Model | Previous stop | Final | Previous selected | Full60 selected | Inner-val S2 BA | Outer S2 BA |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | EEGNet | 22 | 60 | 14 | 50 | 0.9331 | 0.9233 |
| 0 | SIRE-EEG | 39 | 60 | 31 | 31 | 0.9282 | 0.9177 |
| 1 | EEGNet | 33 | 60 | 25 | 25 | 0.9408 | 0.9161 |
| 1 | SIRE-EEG | 30 | 60 | 22 | 22 | 0.9496 | 0.9057 |
| 2 | EEGNet | 39 | 60 | 31 | 31 | 0.9193 | 0.9269 |
| 2 | SIRE-EEG | 27 | 60 | 19 | 43 | 0.9241 | 0.9194 |
| 3 | EEGNet | 32 | 60 | 24 | 39 | 0.9122 | 0.9219 |
| 3 | SIRE-EEG | 32 | 60 | 24 | 60 | 0.9063 | 0.9195 |
| 4 | EEGNet | 18 | 60 | 10 | 59 | 0.9486 | 0.9175 |
| 4 | SIRE-EEG | 34 | 60 | 26 | 54 | 0.9448 | 0.9254 |
