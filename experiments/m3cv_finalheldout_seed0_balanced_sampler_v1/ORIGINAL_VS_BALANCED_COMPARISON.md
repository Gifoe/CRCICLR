# M3CV outer-development: original trial-equal versus hierarchical balanced sampling

All rows use the same frozen seed-0 development folds. The new run changes only the sampler and does not access final-heldout arrays.

| Setting | Model | S2 BA | S2 Macro-F1 | WS-BA |
|---|---|---:|---:|---:|
| Original trial-equal | EEGNet | 0.9130 | 0.9119 | 0.8816 |
| Original trial-equal | SIRE-EEG | 0.9105 | 0.9092 | 0.8790 |
| Hierarchical balanced | EEGNet | 0.9155 | 0.9145 | 0.8821 |
| Hierarchical balanced | SIRE-EEG | 0.9121 | 0.9108 | 0.8796 |

| Setting | SIRE-EEG minus EEGNet | S2 BA | 95% CI | WS-BA | 95% CI |
|---|---|---:|---:|---:|---:|
| Original trial-equal | paired subject-level | -0.0025 | [-0.0085, +0.0034] | -0.0026 | [-0.0092, +0.0041] |
| Hierarchical balanced | paired subject-level | -0.0035 | [-0.0090, +0.0023] | -0.0025 | [-0.0080, +0.0032] |

Balanced sampling improves SIRE relative to EEGNet on S2 BA: `False`.
Balanced sampling improves SIRE relative to EEGNet on WS-BA: `True`.
