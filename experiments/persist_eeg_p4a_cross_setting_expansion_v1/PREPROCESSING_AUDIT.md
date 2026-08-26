# Preprocessing Audit

WBCIC S4 inherits the frozen P3 cache and preprocessing. OpenBMI ERP S5/S6 inherit the Stage-0 manifest pipeline (1–45 Hz, 250 Hz, 0–1 s, 62 channels, no baseline correction). Runtime float16 is a storage-only cast; model-fit-only channel normalization is applied during training.
