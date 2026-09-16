# EEG Conformer and FBCNet final baselines

Four frozen tasks, five folds and three seeds for each model (120 training cells). Training, checkpoints, fixed filter-bank scratch and heldout per-session runtime records stay outside Git. The code refuses final true-heldout inference until all 120 checkpoints exist and a SHA-256 pre-evaluation lock has been written. The final result artifacts are generated only after all-session heldout inference and completeness validation.

The official SIRE-EEG is a LiteBN from another server. This experiment never loads its checkpoint. If its subject-level true-heldout rows are unavailable, the paired comparison is explicitly unavailable rather than silently comparing against this server's different LiteBN.
