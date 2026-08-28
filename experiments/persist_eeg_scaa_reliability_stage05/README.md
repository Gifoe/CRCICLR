# PERSIST-EEG SCAA Reliability Stage-0.5

This development-only certificate reliability audit asks whether S1/S2-observable temporal stability predicts whether a subject/backbone S2 adaptation certificate persists to S3. It preserves the completed Stage-0 terminal `TARGET_HISTORY_UTILITY_TRANSFER_PARTIAL` and does not build SCAA.

The experiment uses only the 41 frozen WBCIC development subjects. Feature extraction reads S1/S2 signal only; S3 enters later solely through the committed Stage-0 utility table as an outcome. The sealed outer 10 remain untouched and unenumerated, and OpenBMI is not accessed.

Run order:

1. `python code/freeze_protocol.py`, then commit the locks and frozen code.
2. `python code/extract_features.py` on the authorized server GPU.
3. `python code/analyze.py`.
4. `python code/plot_publication.py`.
5. `python code/validate.py`.

Runtime caches are intentionally excluded from Git. Compact code, locks, tables, reports, statistics, and figures are committed.

