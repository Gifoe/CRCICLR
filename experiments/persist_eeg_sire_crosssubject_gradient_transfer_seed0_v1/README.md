# Cross-subject SIRE gradient transfer (seed 0)

This is a **one-step diagnostic**, not a training run. Within each of five outer folds, inner-train subjects are partitioned into five disjoint meta-validation groups. At the canonical checkpoint, separate full-CE, Shared2 P-only utility, and explicit Shared1 Pearson-persistence gradients are computed in the four-weight `SCOPE_B_SHARED` space. Raw, norm-balanced, and conflict-projected combinations are compared at three equal parameter-space displacements.

The code loads the exact frozen Shared1/Shared2 geometry from the completed layerwise audit and checks its hashes. It evaluates virtual effects on source and held subjects, restores the full canonical model state after **every** candidate, and freezes all inner-train analysis and code hashes before any discovery read. Discovery uses only the predeclared primary step and six directions. Outer-dev and final-heldout EEG are never read.

The first discovery read stopped before any candidate evaluation because its extraction lacked frozen-reference P/C cache fields. The cache-only fix and amended code hash are recorded in `protocol/META_ANALYSIS_FREEZE_AMENDMENT.json`; inner-meta results, virtual directions, step sizes and decision rules were unchanged.

Run in order on the server with canonical data assets and the prior layerwise runtime:

```bash
for f in 0 1 2 3 4; do python code/run.py --meta-fold "$f"; done
python code/run.py --freeze-meta
for f in 0 1 2 3 4; do python code/run.py --discovery-fold "$f"; done
python code/run.py --aggregate
```

Commands above assume the working directory is this experiment directory; `code/run.py` also works from the repository root with its full path. Runtime intermediates stay under `SIRE_GRADIENT_TRANSFER_RUNTIME`, outside Git. Commit only compact protocol, code, tables, audits and report.
