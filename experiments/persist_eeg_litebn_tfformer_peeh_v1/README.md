# LiteBN / TFFormer PEEH v1

This experiment applies the manuscript's frozen Protected-coordinate diagnostic to LiteBN and TFFormer across OpenBMI MI, ERP, SSVEP, and WBCIC MI (five folds and seeds 0, 1, 2). It does not retrain either neural model or update BatchNorm state.

Run from the repository root:

```bash
python experiments/persist_eeg_litebn_tfformer_peeh_v1/code/run_peeh.py
```

The runner caches 64-dimensional pre-classifier embeddings and resumable per-run JSON files under `/root/rivermind-data/litebn_tfformer_peeh_v1_runtime`. Repository outputs are written to `outputs/peeh_v1/`. The protocol lock records checkpoint hashes, split provenance, evaluation cohorts, and the exact selection/erasure conventions.

The primary quantity is `PEEH = BA_random_erased - BA_protected_erased`, reported in percentage points after averaging fold/seed repetitions within each biological subject and bootstrapping subjects 20,000 times. Positive PEEH supports predictive consequence of TRAIN-selected persistent coordinates; it is not by itself a model-superiority metric.
