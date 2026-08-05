# Reproduce the contextual-risk screening

```bash
cd /root/autodl-tmp/hsc_tta_eeg/repo
conda activate hsc_gpu
python -m pip install -e .
PYTHONPATH=src python scripts/contextual_risk/run_all.py \
  --project-root /root/autodl-tmp/hsc_tta_eeg \
  --device cuda
```

The run reuses validated per-subject source caches when their schema, episode hash, and source-model hash match. It then rebuilds the shared feature/surface tables, writes the screening freeze before any OOF outer result, executes both five-rotation development-only branches, and applies the frozen gate. The expected scientific terminal decision for this artifact is `STOP_CONTEXTUAL_RISK_ALLOCATION`.

Run the complete tests with:

```bash
PYTHONPATH=src python -m pytest -q
```

Raw EEG, token HDF5 files, checkpoints, and per-subject source-cache NPZ files are intentionally excluded from Git. Aggregate and subject-level screening tables, manifests, validation outputs, and reports are included.
