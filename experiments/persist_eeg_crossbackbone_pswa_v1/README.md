# Cross-backbone PSWA v1

This is a frozen-artifact, probe-only audit. It never trains a model, reruns
inference, recomputes Protected assignments, recomputes the persistence
spectrum, or resamples random coordinate controls.

The initial runner verified whether the corrected PEEH run persisted every
artifact needed to estimate Protected-only and equal-rank random-only WS-BA.
The authorized repair adds deterministic recovery of only the canonical
coordinate transform plus one frozen inference pass. It reads the final
Protected indices from corrected PEEH JSON and never reruns PEEH selection,
persistence permutations, utility splits, or utility erasures.

`run_pswa_recovery.py` writes compact, resume-safe canonical-q caches outside
Git. `run_priority3_recovery.ps1` isolates every cell in a fresh Python process
to avoid the Windows CUDA/NumPy state-corruption failures observed in long-lived
workers.
