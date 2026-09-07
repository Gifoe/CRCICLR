# Smoke download report

Initial readable smoke payloads were EEGMMIDB subject 001 target runs, HMC SN001 with scoring sidecars, and CAP brux1 with TXT/ST sidecars. Smoke reads passed before full download. Full-download verification supersedes the smoke status:

| dataset | manifest files | SHA verified | missing/failed |
| --- | --- | --- | --- |
| eegmmidb | 654 | 654 | 0 |
| hmc | 453 | 453 | 0 |
| cap | 324 | 324 | 0 |

Source URLs are the public PhysioNet endpoints recorded in `configs/datasets/*.yaml`. No authenticated or foundation-model download was used.
