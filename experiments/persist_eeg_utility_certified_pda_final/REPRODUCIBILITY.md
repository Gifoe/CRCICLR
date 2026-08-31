# Reproducibility

Run from the repository root with the authorized environment:

```text
set PYTHONPATH=experiments/persist_eeg_utility_certified_pda_final/code
python experiments/persist_eeg_utility_certified_pda_final/code/run_source.py --datasets OpenBMI WBCIC --backbone ATCNet-CleanRoom
python experiments/persist_eeg_utility_certified_pda_final/code/make_package.py
python experiments/persist_eeg_utility_certified_pda_final/code/validate.py
```

The source archives are referenced by path but not copied into this package. Runtime/checkpoint/cache/raw EEG files are excluded by `.gitignore`.
