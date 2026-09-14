# LiteBN-TFFormer four-task multiseed diagnostic

This experiment extends the frozen LiteBN-TFFormer architecture to seed1 and
seed2 on OpenBMI SSVEP, OpenBMI ERP, OpenBMI MI, and WBCIC MI. Each task uses
the matched historical LiteBN checkpoint from the same seed. Checkpoints are
selected on five-fold subject-equal inner validation, and internal heldout is
evaluated only after all five checkpoints for a task are frozen.

Seed1 and seed2 use a minimum of 10 epochs and patience 8 without strict
inner-validation BA improvement. The historical seed0 SSVEP result used the
original full-60-epoch schedule, so the combined three-seed table is explicitly
reported as a nonuniform-schedule internal diagnostic rather than a strict
uniform-protocol estimate.

Runtime checkpoints are intentionally excluded from Git. Provenance hashes,
selection tables, trajectories, heldout subject rows, aggregate tables, and
the terminal report are retained.
