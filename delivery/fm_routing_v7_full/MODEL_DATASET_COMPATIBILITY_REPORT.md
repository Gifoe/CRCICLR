# Model–dataset compatibility report

Protocol freeze: `c82564b54b288e648783c5e152a6fa5551433ada570d6a509f74248d0246ef61`.

## Admissible pairs

| Model | Datasets that pass all A-COMP checks |
|---|---|
| cbramod | hmc, eegmmidb |
| eegpt | none |
| bendr | none |
| brant | none |
| eeg2rep | none |
| neurogpt | none |
| biot | eegmmidb |
| labram | eegmmidb |

## Pre-registered combinations

| Priority | Datasets | Shared admissible models | Pass |
|---:|---|---|---|
| 1 | hmc + eegmmidb | cbramod | False |
| 2 | hmc + bcic2a | none | False |
| 3 | sleepedffull + eegmmidb | none | False |
| 4 | sleepedffull + bcic2a | none | False |

HMC contains actual `C3-M2` and `C4-M1` bipolar signals. They were not relabeled as C3/C4 or C3-P3/C4-P4. SleepEDFFull is absent. Consequently no pre-registered cross-task pair has three jointly admissible model families.

No label value, prediction, or task-performance metric was used in this selection.
