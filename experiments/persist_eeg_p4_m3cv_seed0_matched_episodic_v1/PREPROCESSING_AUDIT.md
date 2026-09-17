# M3CV preprocessing audit

No new signal preprocessing was performed. The cache reads every valid native 4-s segment directly from BrainVision: no resampling, filtering, rereferencing, cropping, artifact rejection, channel reordering, duplication, truncation or cache normalization.

- Source SoftwareFilters (recorded, not applied by this runner): `[{"BandpassFilter": {"highFreq": 200, "lowFreq": 0.01, "order": 4, "phase": "zero-phase", "rolloff": "24 dB/octave", "type": "Butterworth"}, "NotchFilter": {"highFreq": 51, "lowFreq": 49, "order": 4, "phase": "zero-phase", "rolloff": "24 dB/octave", "type": "Butterworth"}}]`.
- Sidecars document linked TP9/TP10 mastoid reference, AFz ground, ICA visual eye-artifact processing, source-level bad-channel interpolation when applicable, and no bad-epoch rejection.
- MNE BrainVision calibration decodes the file into physical values; it is format decoding, not a newly introduced filter or normalization.

## Eligible-cache valid-trial distributions
- `ses-01/left_hand`: min=80, median=80.0, mean=80.16, max=88 (N subjects=93).
- `ses-01/right_hand`: min=79, median=80.0, mean=80.14, max=90 (N subjects=93).
- `ses-02/left_hand`: min=9, median=43.0, mean=45.18, max=98 (N subjects=93).
- `ses-02/right_hand`: min=11, median=41.0, mean=45.14, max=99 (N subjects=93).
