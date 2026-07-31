# CPU data schema

Recording manifests include dataset-prefixed subject IDs, recording IDs, path/size, duration, sampling rate, channel names/count, annotation provenance/labels/count, optional start time, readability, integrity, eligibility, and explicit exclusion reason.

Subject HDF5 caches contain signal, label, time boundaries, channel names/mask, sampling rate, recording/run IDs, quality flags, a configuration hash, and an atomic completion marker. Raw data are never modified. Episode rows carry separate context and future indices/recordings/runs plus counts, durations, protocol, role, seed, and exclusion reason.

