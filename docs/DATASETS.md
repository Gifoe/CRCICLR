# Datasets

Only official public PhysioNet releases are used: HMC Sleep Staging 1.1, CAP Sleep Database 1.0.0, and EEGMMIDB 1.0.0. Sleep labels map to Wake/N1/N2/N3/REM; CAP S3/S4 merge into N3 and movement/unscored labels are excluded. EEGMMIDB uses only runs 4, 6, 8, 10, 12, and 14; T1/T2 semantics depend on run parity group.

Record indexes come from `physionet.org/files`; payloads may use the official `physionet-open.s3.amazonaws.com` open-data bucket when the website endpoint is throttled. Manifests preserve the exact payload URL and local SHA256.

CAP filenames are conservatively treated as stable recording-level subject identifiers unless authoritative metadata proves repeated recordings belong to the same person. This prevents accidental merging; the audit report must disclose the consequence.
