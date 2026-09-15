# LiteBN / TFFormer PEEH v1

This experiment applies the manuscript's frozen Protected-coordinate diagnostic to LiteBN and TFFormer across OpenBMI MI, ERP, SSVEP, and WBCIC MI (five folds and seeds 0, 1, 2). It does not retrain either neural model or update BatchNorm state.

Appendix-L repair from frozen embeddings (the command fails rather than falling
back to neural inference if any cache is missing):

```bash
python experiments/persist_eeg_litebn_tfformer_peeh_v1/code/run_peeh.py --cached-only
```

The runner caches 64-dimensional pre-classifier embeddings and resumable per-run JSON files under `/root/rivermind-data/litebn_tfformer_peeh_v1_runtime`. Repository outputs are written to `outputs/peeh_v1/`. The protocol lock records checkpoint hashes, split provenance, evaluation cohorts, and the exact selection/erasure conventions.

The primary quantity is `PEEH = BA_random_erased - BA_protected_erased`, reported in percentage points after averaging fold/seed repetitions within each biological subject and bootstrapping subjects 20,000 times. All selection and final erasure ridge probes use the full 64-dimensional representation. Positive PEEH supports predictive consequence of TRAIN-selected persistent coordinates; it is not by itself a model-superiority metric.

WBCIC is labeled **source-session persistence -> future-session consequence**:
Protected persistence is estimated from source sessions S0/S1, while consequence
is evaluated on the already-accessed true-outer S2 cohort. Task summaries also
report Protected assignment coverage as non-empty fold-seed runs out of 15.
