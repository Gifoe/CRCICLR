# SIRE-EEG P3 diagnostic/design bridge

This package contains two separate, frozen analyses. Historical source names
`LiteBN` and `CompactLite` denote checkpoint provenance, not the manuscript
model name.

Run on the existing Linux server repository, in order:

```bash
python experiments/persist_eeg_sire_diagnostic_design_bridge_v1/code/source_audit.py
python experiments/persist_eeg_sire_diagnostic_design_bridge_v1/code/run_subspace_decomposition.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/persist_eeg_sire_diagnostic_design_bridge_v1/code/prepare_and_freeze_selection.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/persist_eeg_sire_diagnostic_design_bridge_v1/code/evaluate_frozen_selection.py
python experiments/persist_eeg_sire_diagnostic_design_bridge_v1/code/finalize_bridge.py
python experiments/persist_eeg_sire_diagnostic_design_bridge_v1/code/verify_bridge.py
```

The prepare phase does not construct an outer-development bundle. The evaluate
phase refuses to run without a SHA-verified `SELECTION_FREEZE.json`. Runtime
embeddings/logs remain on the server and are not part of the committed outputs.

At the user's direction, Part B reuses the historical frozen Full checkpoint
as B0 and does not retrain. That checkpoint is **not** training-protocol
matched to B1/B2/B4, and historical Full outer-development results had already
been computed. Part B is therefore a pre-specified development replay, not an
independent prospective test. The selector must remain unchanged for a future
P4 confirmation cohort.
