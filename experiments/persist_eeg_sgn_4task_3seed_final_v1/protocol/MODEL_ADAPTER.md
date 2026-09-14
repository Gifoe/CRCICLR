# Model adapter and compatibility repair

The cache boundary `B,C,T` is transposed to the official SGN `B,T,C` classification boundary. No electrode coordinates, montage graph, filtering, resampling, augmentation, calibration, or task-specific architecture tuning is used.

The release imports itself through the absent package prefix `SGN4MTSC` and hard-codes a TDBRAIN-only `groups_matrix_T.npy` with 33 variables. Imports were made package-local. For arbitrary EEG channel count, assignment logits use the same official one-hot plus clipped Gaussian-noise initialization family, with deterministic balanced variable indices and the cell's frozen seed. This changes no VGE, MGWM or PWSM operation and reads no validation, outer-development, or heldout data.

