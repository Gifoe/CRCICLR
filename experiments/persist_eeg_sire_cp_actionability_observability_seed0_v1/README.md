# SIRE-EEG C-direction actionability and observability

This experiment tests whether the oracle-headroom / low-router-recovery pattern previously measured on EEGNet changes on the frozen canonical SIRE-EEG seed-zero checkpoints. It runs OpenBMI MI and SSVEP, folds 0–4, and does not train or update any neural parameter.

The source stage is the 48-channel branch concatenation before `depth1`. The successor is the first shared block after `depth1 -> point1 -> norm1 -> ELU -> pool`, before `depth2`. Stage projectors are fitted from inner-train subject-session-label centroids by the mechanism-closure standardized kernel-ridge PathFit, using only the frozen SIRE PERSIST/PEEH canonical protected coordinates as targets. Complement PCA, router compression, and utility ranking are fold local and use inner-train only.

Run `python code/run.py lock` before any cell, then run `python code/run.py cell --task OpenBMI_MI --fold 0` for a single-cell identity probe. After the probe passes, run all ten cells with `python code/run.py all`; `aggregate` can resume final collation once all cell receipts exist. Runtime caches are stored outside the repository in `sire_cp_actionability_seed0_runtime`.

Only inner-train and discovery EEG rows are loaded. Outer-development rows and OpenBMI's 14 final-heldout EEG arrays are excluded. The source audit still discloses that the reused historical checkpoints had earlier final-heldout diagnostic evaluations.

The compressed full response-curve archive is a formal result. Per-cell activations, router arrays, and log files are runtime-only and are not part of the GitHub result bundle.
