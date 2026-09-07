# Full CPU preprocessing report

Generated: 2026-08-01T03:49:35.849540+00:00

| dataset | complete caches | failures | windows | one-channel caches | size |
| --- | --- | --- | --- | --- | --- |
| eegmmidb | 109 | 0 | 9837 | 0 | 1.40 GiB |
| hmc | 151 | 0 | 137243 | 0 | 5.73 GiB |
| cap | 103 | 0 | 103021 | 102 | 2.17 GiB |

Configuration: sleep uses 30 s non-overlapping epochs, 0.3–40 Hz band-pass, 200 Hz target rate, and no default notch; EEGMMIDB uses runs 4/6/8/10/12/14, 1–40 Hz, 160 Hz, and up to 4 s per mapped imagery event. Caches store physical signal, labels, time boundaries, recording/run IDs, channel names/mask, sampling rate, three per-window quality flags, and the preprocessing config hash.

The container cgroup memory limit is 2147483648 bytes. HMC and CAP therefore ran serially; CAP used one fresh Python process per remaining subject after long-process allocator pressure was observed. Resume checks occur before raw EDF loading. All final caches have `complete=true`; failed subject count is zero.

Mapped label/event counts:

| dataset | windows | mapped label counts |
| --- | --- | --- |
| eegmmidb | 9837 | {0: 2479, 1: 2438, 2: 2465, 3: 2455} |
| hmc | 137243 | {0: 23686, 1: 15548, 2: 50083, 3: 26671, 4: 21255} |
| cap | 103021 | {0: 18615, 1: 4551, 2: 37223, 3: 24559, 4: 18073} |
