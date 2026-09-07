# PERSIST-EEG External Actionability V1

Prospective EEGMMIDB repeated-run audit following frozen DDA-V1.  Run order:

1. `freeze_protocol.py`
2. `extract_features.py inventory`
3. `extract_features.py extract`
4. `external_actionability_v1.py train`
5. `external_actionability_v1.py embed`
6. `external_actionability_v1.py discover`
7. `external_actionability_v1.py audit`
8. `external_actionability_v1.py report`

The sealed outer split is not opened by the extractor or audit program.
