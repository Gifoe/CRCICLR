# EEGNet P2 selection-factor ablation

This experiment compares Random, Persistence-only, Utility-only, and the exact
stored corrected-PEEH Joint/Protected union at identical rank.  It uses frozen
EEGNet seed-0 checkpoints and the corrected PSWA retained-subspace probe.

No neural model is trained or fine-tuned.  Selector construction uses only
TRAIN-subject quantities.  Final-heldout q/y arrays are opened only after the
per-cell selector freeze object and its SHA-256 have been written.

The corrected PEEH cell artifacts serialized the persistence gate but omitted
the numerical persistence margin.  The runner reconstructs only that missing
deterministic TRAIN-only statistic with the original frozen spectrum code path
and asserts that every reconstructed gate equals the stored corrected gate.
Joint itself is never reconstructed or reselected; the stored Protected union
is the source of truth.
