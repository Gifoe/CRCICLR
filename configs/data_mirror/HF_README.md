---
license: other
pretty_name: PERSIST-EEG Stage-0 public data mirror
---

# PERSIST-EEG Stage-0 public data mirror

Public, byte-preserving research mirror for PERSIST-EEG Stage-0. This repository is a mirror, not a newly collected dataset; it does not impose one new repository-wide license. Each subdataset retains its applicable upstream license and notices.

Included upstream datasets:

- Lee et al. 2019 OpenBMI Motor Imagery, ERP/P300, and SSVEP: 54 participants and two sessions; original GigaDB dataset 100542, DOI `10.5524/100542`, released under CC0. When present, files are NEMAR BIDS-formatted derivatives (`nm000338`, `nm000323`, `nm000273`), not the untouched GigaDB archive. NEMAR version, derivative details, and upstream format/provenance are in `SOURCE_PROVENANCE.md`.
- EEG Motor Movement/Imagery Dataset v1.0.0: 109 participants, PhysioNet source, DOI `10.13026/C28G6P`, Open Data Commons Attribution License v1.0. Required attribution and notice are retained in `LICENSE_AUDIT.md`.

The mirror performs no filtering, resampling, rereferencing, normalization, epoching, channel selection, artifact rejection, feature extraction, or label alteration. OpenBMI's NEMAR distributions are BIDS-formatted derivatives and are identified as such in `SOURCE_PROVENANCE.md`.

Licensing and attribution are documented in `LICENSE_AUDIT.md`. Per-file SHA256, source URLs, retrieval dates, transfer state, completeness tables, and verification results are under `metadata/`.
