# M3CV final-heldout three-seed logit ensemble

Each result first averages raw logits over five frozen development-fold checkpoints for the same seed; it then averages the three seed-level values within each biological heldout subject.

| Model | S2 BA | S2 Macro-F1 | WS-BA |
|---|---:|---:|---:|
| EEGNet | 92.90% | 92.77% | 89.75% |
| SIRE-EEG | 93.04% | 92.92% | 89.26% |

| SIRE-EEG minus EEGNet | Mean | 95% subject bootstrap CI |
|---|---:|---:|
| S2_BA | +0.14 pp | [-0.71, +0.91] pp |
| WS_BA | -0.49 pp | [-1.71, +0.57] pp |

| Model | PEEH | PEEH coverage | PSWA | PSWA coverage |
|---|---:|---:|---:|---:|
| EEGNet | +6.66 pp [+5.59, +7.76] | 15/15 | +8.80 pp [+7.97, +9.66] | 15/15 |
| SIRE-EEG | +5.51 pp [+4.72, +6.37] | 15/15 | +8.43 pp [+7.51, +9.26] | 15/15 |

PEEH empty Protected assignments are an exactly-zero effect. PSWA empty Protected assignments are undefined and omitted; coverage is nonempty fold/checkpoint runs out of 15, never a subject count.
