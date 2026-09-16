# P4 seed-0 protocol

Shin2017B / nm000268 uses only the 30 EEG channels and the three mental-arithmetic sessions (S1=1arithmetic, S2=3arithmetic, S3=5arithmetic). Whole 10-second task trials are used once; no MI, NIRS, EOG, pseudo-trials, calibration, adaptation, or outer-subject information is used.

Outer CV is deterministic 5-fold subject-disjoint (seed 0); each fold reserves five remaining subjects for S3 inner validation. Training uses TRAIN S1+S2. Channel-wise mean/std is fit only on TRAIN S1+S2. The fixed recipe is 60 epochs, min selection epoch 10, AdamW lr=3e-4, weight decay=5e-4, batch 128, clip=5. Models are frozen EEGNet and current final LiteBN, reported as SIRE-EEG; only C=30, T=2000, and binary head are adapted.

Primary full-model endpoint is outer S3. BA and macro-F1 are subject-equal; WS-BA=min(S1,S2,S3). PEEH/PSWA reuse the frozen PERSIST implementation: 200 permutations, active-rank/eigengap/block rules, 100 equal-rank controls, ridge alpha .01, 20,000 biological-subject bootstrap draws. Coordinates and Protected selection use TRAIN S1/S2 only; PEEH probe fits TRAIN S1 only as in the existing implementation and evaluates outer S3. PSWA probes use TRAIN S1/S2 and evaluate outer S1/S2/S3.
