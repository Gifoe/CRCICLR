# Seed-0 replay numerical environment amendment

The formal PEEH source text is identical to the published worktree after line-ending normalization. The current server uses PyTorch 2.8.0+cu128 and NumPy 2.2.6 on an RTX 5090. With multiple OpenBLAS threads, Python intermittently exits `0xc0000005` in NumPy `_multiarray_umath`; with one thread, the GPU path runs.

The same EEGNet/OpenBMI-MI/fold1 cell reproduced exactly on GPU with one thread. On CBraMod/OpenBMI-MI/fold1, frozen checkpoint SHA, active rank, Protected coordinates, gate decisions, intact BA, and Protected-erased BA reproduced exactly. The mean of 100 random-erasure controls differed by at most 0.046875 percentage points across subjects, corresponding to a few near-tie trial predictions among the 100 controls. Two- and three-thread BLAS reruns reduced but did not eliminate the discrepancy; four threads crashed natively. A CPU fallback changed Protected-erased BA and is rejected.

Before any new selector heldout evaluation, replay acceptance is fixed as:

- exact checkpoint SHA, active rank, candidate block identities, Protected coordinates, and gate decisions;
- TRAIN utility-evidence scalar difference at most `1e-4` cross-entropy units;
- intact and Protected-erased subject BA difference at most `1e-10` BA;
- 100-control mean random-erased subject BA difference at most `0.001` BA = `0.1` percentage point.

This tolerance applies only to the 100-control random mean. It does not alter the selector, coordinate construction, statistical protocol, or any heldout optimization rule. If a cell exceeds any bound, new selector analysis remains gated off. Every difference is retained in `SEED0_REPLAY_AUDIT.csv`.
