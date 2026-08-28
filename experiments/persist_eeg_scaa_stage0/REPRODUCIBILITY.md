# Reproducibility

Run from repository root with the frozen server environment:

1. `D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe experiments\persist_eeg_scaa_stage0\code\run_utility.py`
2. `D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe experiments\persist_eeg_scaa_stage0\code\analyze.py`
3. `D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe experiments\persist_eeg_scaa_stage0\code\validate.py`

The protocol lock records code, data-lock, recipe, checkpoint, and normalizer hashes. Primary inference uses 10,000 subject resamples; seeds are averaged within subject and are not treated as biological replicates. Runtime trial predictions and raw EEG are excluded from Git.
