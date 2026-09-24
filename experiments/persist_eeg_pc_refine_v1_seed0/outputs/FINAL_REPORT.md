# Post-heldout technical recovery

The locked V1 aggregation failed because fold-specific Protected ranks differ. This deterministic recovery changes only the post-hoc top-three correction-energy summary. The primary BA/F1/NLL calculations, locked predictions, and checkpoints are unchanged. Per the protocol, results produced by this post-heldout code repair are labeled exploratory; they do not overwrite a confirmatory V1 result.

# PC-Refine EEGNet V1 final report

Seed 0; five-fold probability mean; formal final-heldout subjects; primary future session.

| Task | Baseline BA | Protected-PC BA | Δ BA vs Baseline |
| --- | ---: | ---: | ---: |
| OpenBMI_MI | 0.7114 | 0.7136 | +0.0021 |
| OpenBMI_ERP | 0.8645 | 0.8624 | -0.0021 |
| OpenBMI_SSVEP | 0.9200 | 0.9214 | +0.0014 |
| WBCIC_MI | 0.7905 | 0.8115 | +0.0210 |

## Paired heldout differences

Biological-subject bootstrap, 20,000 draws. Units are absolute metric points.

| Task | Metric | Difference | 95% CI |
| --- | --- | ---: | ---: |
| OpenBMI_MI | BA | +0.0021 | [-0.0021, +0.0064] |
| OpenBMI_MI | macro_F1 | +0.0022 | [-0.0024, +0.0073] |
| OpenBMI_MI | worst_session_BA | +0.0000 | [-0.0057, +0.0057] |
| OpenBMI_ERP | BA | -0.0021 | [-0.0045, -0.0003] |
| OpenBMI_ERP | macro_F1 | +0.0046 | [+0.0019, +0.0071] |
| OpenBMI_ERP | worst_session_BA | -0.0013 | [-0.0034, +0.0008] |
| OpenBMI_SSVEP | BA | +0.0014 | [+0.0000, +0.0036] |
| OpenBMI_SSVEP | macro_F1 | +0.0016 | [+0.0000, +0.0040] |
| OpenBMI_SSVEP | worst_session_BA | +0.0007 | [+0.0000, +0.0021] |
| WBCIC_MI | BA | +0.0210 | [+0.0075, +0.0360] |
| WBCIC_MI | macro_F1 | +0.0344 | [+0.0161, +0.0539] |
| WBCIC_MI | worst_session_BA | +0.0225 | [+0.0070, +0.0370] |

## Physical-session subject-equal BA

OpenBMI physical S1/S2; WBCIC physical S0/S1/S2 corresponds to paper S1/S2/S3.

| Task | Session | Baseline BA | Protected-PC BA |
| --- | --- | ---: | ---: |
| OpenBMI_MI | S1 | 0.7143 | 0.7157 |
| OpenBMI_MI | S2 | 0.7114 | 0.7136 |
| OpenBMI_ERP | S1 | 0.8633 | 0.8623 |
| OpenBMI_ERP | S2 | 0.8645 | 0.8624 |
| OpenBMI_SSVEP | S1 | 0.9493 | 0.9493 |
| OpenBMI_SSVEP | S2 | 0.9200 | 0.9214 |
| WBCIC_MI | S0 | 0.7780 | 0.8035 |
| WBCIC_MI | S1 | 0.7815 | 0.8185 |
| WBCIC_MI | S2 | 0.7905 | 0.8115 |

## Interpretation

Protected-PC has a positive heldout BA difference on 3/4 tasks.
The user narrowed this run to the modified model and its matched seed-0 baseline. This comparison cannot isolate a mechanism-specific gain from a generic adapter effect.
Random-PC is a separate exploratory follow-up only for tasks with a positive primary heldout BA difference.
