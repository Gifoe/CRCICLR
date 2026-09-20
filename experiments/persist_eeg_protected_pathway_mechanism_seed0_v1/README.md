# Frozen final-Protected pathway mechanism audit

This seed-0 audit traces only the final classifier-input Protected coordinates
locked by `persist_eeg_native_protected_utilization_seed0_v1`.  It uses
OpenBMI MI/SSVEP outer-development subjects, five frozen folds, and the exact
existing checkpoints/normalizers/bases/assignments.  No model, native head, or
task probe is trained or refit; the only fitted objects are TRAIN-only linear
ridge mappings from an architecture stage to the fixed final coordinates.

Outputs are compact CSV/JSON/Markdown only. Runtime activations and pathways
remain outside Git. Final/true-heldout subjects are never opened.
