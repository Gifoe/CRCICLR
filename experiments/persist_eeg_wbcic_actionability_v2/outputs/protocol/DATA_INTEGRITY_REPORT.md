# WBCIC/Yang2025 data integrity report

Terminal state: `WBCIC_CORE_DATA_INTEGRITY_PASS`

- Core subjects: 51
- Sessions/BDF: 153
- Manifest files: 1234
- Manifest bytes: 68048414654
- All 1,234 local files passed their frozen NEMAR manifest content checksum (SHA-256 for BDF; Git blob SHA-1 otherwise).
- Every BDF header opened at 1000 Hz with 64 total channels.
- Every session has 59 EEG channels including Pz, left/right events, and no malformed event rows.
- No sourcedata file is present.
