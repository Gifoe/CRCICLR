# BNCI2015-001 data audit

- Dataset: BNCI2015-001 / NEMAR nm000140; source: MOABB official BNCI downloader and its _convert_mi MAT-to-Raw converter; only explicit A/B URLs were requested.
- MOABB version: `1.7.2`; official source URL prefix: `https://lampx.tugraz.at/~bci/database/001-2015/`.
- Expected/observed cohort: 12 subjects, all with common `0A` and `1B`. The public loader lists `2C` only for sub-08, sub-09, sub-10 and sub-11; it was deliberately excluded before downloading.
- Native source: 13 EEG channels at 512 Hz; ordered `['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4']`.
- Source event map: `right_hand=1`, `feet=2`; cached y map: right_hand=0, feet=1.
- Source preprocessing recorded by MOABB metadata: 0.5--100 Hz bandpass, 50 Hz notch, CAR. This replication adds no dataset-specific frequency filter or CSP.
- Deterministic resampling: MNE Raw.resample(250 Hz, npad='auto'); fixed interval [event+1.25s,event+5.25s) yields exactly T=1000.
- Official raw A/B downloads: 24 MAT files, 1,567,974,212 bytes. Session-C files: zero.
- Prepared cache: 231,808,160 compressed bytes, 24 cells; each X=[200,13,1000] float32 and y=[200] int64.

| Subject | Available public sessions | Used session | Trials | right_hand | feet |
|---|---|---|---:|---:|---:|
| sub-01 | 0A, 1B | 0A | 200 | 100 | 100 |
| sub-01 | 0A, 1B | 1B | 200 | 100 | 100 |
| sub-02 | 0A, 1B | 0A | 200 | 100 | 100 |
| sub-02 | 0A, 1B | 1B | 200 | 100 | 100 |
| sub-03 | 0A, 1B | 0A | 200 | 100 | 100 |
| sub-03 | 0A, 1B | 1B | 200 | 100 | 100 |
| sub-04 | 0A, 1B | 0A | 200 | 100 | 100 |
| sub-04 | 0A, 1B | 1B | 200 | 100 | 100 |
| sub-05 | 0A, 1B | 0A | 200 | 100 | 100 |
| sub-05 | 0A, 1B | 1B | 200 | 100 | 100 |
| sub-06 | 0A, 1B | 0A | 200 | 100 | 100 |
| sub-06 | 0A, 1B | 1B | 200 | 100 | 100 |
| sub-07 | 0A, 1B | 0A | 200 | 100 | 100 |
| sub-07 | 0A, 1B | 1B | 200 | 100 | 100 |
| sub-08 | 0A, 1B, 2C | 0A | 200 | 100 | 100 |
| sub-08 | 0A, 1B, 2C | 1B | 200 | 100 | 100 |
| sub-09 | 0A, 1B, 2C | 0A | 200 | 100 | 100 |
| sub-09 | 0A, 1B, 2C | 1B | 200 | 100 | 100 |
| sub-10 | 0A, 1B, 2C | 0A | 200 | 100 | 100 |
| sub-10 | 0A, 1B, 2C | 1B | 200 | 100 | 100 |
| sub-11 | 0A, 1B, 2C | 0A | 200 | 100 | 100 |
| sub-11 | 0A, 1B, 2C | 1B | 200 | 100 | 100 |
| sub-12 | 0A, 1B | 0A | 200 | 100 | 100 |
| sub-12 | 0A, 1B | 1B | 200 | 100 | 100 |

Missing/corrupt A/B recordings: none.
