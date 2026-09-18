# M3CV final-heldout seed-0 result

Final-heldout reporting unit: each of 18 subjects after averaging that subject's two-session metric separately across all 5 frozen fold checkpoints.

| Model | S2 BA | S2 Macro-F1 | WS-BA |
|---|---:|---:|---:|
| EEGNet | 0.9256 | 0.9245 | 0.8911 |
| SIRE-EEG | 0.9232 | 0.9220 | 0.8837 |

| SIRE-EEG minus EEGNet | Mean | 95% subject bootstrap CI |
|---|---:|---:
| future BA | -0.0024 | [-0.0117, +0.0068] |
| WS-BA | -0.0074 | [-0.0198, +0.0044] |

No seed 1/2 was run.
