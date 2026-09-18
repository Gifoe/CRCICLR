# M3CV hierarchical balanced-sampler seed-0 result

Development-only result: final-heldout data were not read or evaluated.

| Model | S2 BA | S2 Macro-F1 | WS-BA |
|---|---:|---:|---:|
| EEGNet | 0.9155 | 0.9145 | 0.8821 |
| SIRE-EEG | 0.9121 | 0.9108 | 0.8796 |

| SIRE-EEG minus EEGNet | Mean | 95% subject bootstrap CI |
|---|---:|---:
| future BA | -0.0035 | [-0.0090, +0.0023] |
| WS-BA | -0.0025 | [-0.0080, +0.0032] |

See `ORIGINAL_VS_BALANCED_COMPARISON.md` and `BALANCED_SAMPLER_AUDIT.json` for the frozen-reference comparison and sampler proof.
