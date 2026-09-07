# Data audit report

Generated: 2026-08-01T03:49:35.849540+00:00

| dataset | recordings | readable | eligible subjects | excluded subjects | raw | processed | audit eligible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eegmmidb | 654 | 654 | 109 | 0 | 1.55 GiB | 1.40 GiB | 109 |
| hmc | 151 | 151 | 151 | 0 | 15.73 GiB | 5.73 GiB | 151 |
| cap | 108 | 108 | 103 | 5 | 40.11 GiB | 2.17 GiB | 103 |

All downloaded payloads passed local SHA256 verification. The five CAP exclusions are retained in `exclusions.parquet`; no quality rule was relaxed.

| dataset | reason | count |
| --- | --- | --- |
| cap | required_channels_missing | 5 |

Annotation vocabulary audit rows: 3663. Full recording, subject, channel, annotation-vocabulary, exclusion, and download manifests are under `data/manifests` outside Git.
