---
license: other
pretty_name: PERSIST-EEG Stage-0 private data mirror
---

# PERSIST-EEG Stage-0 private data mirror

Private, byte-preserving research mirror for PERSIST-EEG Stage-0.

Included upstream datasets:

- Lee et al. 2019 OpenBMI Motor Imagery, ERP/P300, and SSVEP, 54 participants and two sessions, from the published NEMAR BIDS distributions (`nm000338`, `nm000323`, `nm000273`), derived from GigaDB DOI `10.5524/100542`.
- EEG Motor Movement/Imagery Dataset v1.0.0, 109 participants, from PhysioNet, DOI `10.13026/C28G6P`.

The mirror performs no filtering, resampling, rereferencing, normalization, epoching, channel selection, artifact rejection, feature extraction, or label alteration. OpenBMI's NEMAR distributions are BIDS-formatted derivatives and are identified as such in `SOURCE_PROVENANCE.md`.

Licensing and attribution are documented in `LICENSE_AUDIT.md`. Per-file SHA256, source URLs, transfer state, completeness tables, and verification results are under `metadata/`.

