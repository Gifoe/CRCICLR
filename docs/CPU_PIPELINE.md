# CPU pipeline

1. Export `CUDA_VISIBLE_DEVICES=""`, audit disk/memory/CPU, and verify `/root/autodl-tmp` is the 350GB data volume.
2. Run tests and synthetic simulations before real data operations.
3. Smoke-download a few official files in EEGMMIDB, HMC, CAP order. Verify readability and manifests.
4. Full download is allowed only when all three smoke checks pass and free space remains at least 300GB. A 60GB guard stops new writes.
5. Audit every discovered recording; exceptions become exclusion records.
6. Preprocess two subjects per dataset before full preprocessing. Raw files are read-only; each subject is one atomic HDF5 cache keyed by configuration hash.
7. Create five deterministic subject-level splits and U/V episodes. Run leakage validators.
8. Emit manifests, CPU report, and the frozen GPU interface. Data, caches, outputs, logs, and state are outside Git.

Long operations run in tmux and write structured logs/state. `.part` identifies incomplete downloads. A completed or hash-matching cache is resumed rather than overwritten.

