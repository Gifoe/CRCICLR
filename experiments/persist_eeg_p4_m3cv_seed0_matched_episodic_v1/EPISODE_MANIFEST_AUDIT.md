# Episode manifest audit

The final `run_stage1.make_manifest` is invoked directly with only an M3CV session-semantic adapter: source support session 1 maps to `ses-01`, and future query session 2 maps to `ses-02`. The stored JSON is metadata-relabeled M3CV but its episode indices derive from the authoritative sampler.

| Fold | Outer | Inner val | Inner train | Legal S1 trials | Steps/epoch | Episodes | Support/query trials | Final manifest SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 19 | 10 | 64 | 10250 | 81 | 4860 | 311040/311040 | `747797fc637252b77f371776e8722e0e265ecae72297891a000266fb8b6b36d9` |
| 1 | 19 | 10 | 64 | 10265 | 81 | 4860 | 311040/311040 | `0ad56163e3266c86ba5ffe5429f5e1d5220003a8c7001b135827438d665c56ad` |
| 2 | 19 | 10 | 64 | 10260 | 81 | 4860 | 311040/311040 | `09b666d7d86fe81cc0b236fcd7358cebfbc60c1ac1fececb8e15b4072d3a23cd` |
| 3 | 18 | 10 | 65 | 10408 | 82 | 4920 | 314880/314880 | `eb50c2e96876dd81ecd175f18dbc60b6075e5da11150a7dd37eef08cf1a16d74` |
| 4 | 18 | 10 | 65 | 10428 | 82 | 4920 | 314880/314880 | `902204ac8a965efe49ac8789c8fe167e0261d8343b49355ef1cb999a5f9ee8fc` |

Each entry was checked: 4 support and 4 different query subjects; all inner-train only; support S1; query S2; 8 LH and 8 RH distinct trials per subject/session/class draw; 64+64 trials. The same files are used for both models.
