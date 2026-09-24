# Native C-to-P transfer gate V2, seed 0

Formal heldout; five-fold probability mean; subject-equal future physical S2 BA.

| Task | Baseline | P-only Gate | Random Gate | Protected Native Gate | Delta vs Base | Delta vs Random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenBMI_MI | 0.7114 | 0.7114 | 0.7121 | 0.7121 | +0.0007 | +0.0000 |
| OpenBMI_ERP | 0.8645 | 0.8641 | 0.8637 | 0.8642 | -0.0003 | +0.0005 |
| OpenBMI_SSVEP | 0.9200 | 0.9200 | 0.9200 | 0.9200 | +0.0000 | +0.0000 |
| WBCIC_MI | 0.7905 | 0.7905 | 0.7910 | 0.7915 | +0.0010 | +0.0005 |

## Paired biological-subject bootstrap

20,000 draws; absolute metric differences.

| Task | Contrast | Metric | Difference | 95% CI |
| --- | --- | --- | ---: | ---: |
| OpenBMI_MI | PROTECTED_NATIVE_TRANSFER_GATE - BASELINE | BA | +0.0007 | [+0.0000, +0.0021] |
| OpenBMI_MI | PROTECTED_NATIVE_TRANSFER_GATE - BASELINE | macro_F1 | +0.0007 | [+0.0000, +0.0022] |
| OpenBMI_MI | PROTECTED_NATIVE_TRANSFER_GATE - BASELINE | worst_session_BA | +0.0007 | [+0.0000, +0.0021] |
| OpenBMI_MI | PROTECTED_NATIVE_TRANSFER_GATE - P_ONLY_TRANSFER_GATE | BA | +0.0007 | [+0.0000, +0.0021] |
| OpenBMI_MI | PROTECTED_NATIVE_TRANSFER_GATE - P_ONLY_TRANSFER_GATE | macro_F1 | +0.0007 | [+0.0000, +0.0021] |
| OpenBMI_MI | PROTECTED_NATIVE_TRANSFER_GATE - P_ONLY_TRANSFER_GATE | worst_session_BA | +0.0000 | [+0.0000, +0.0000] |
| OpenBMI_MI | PROTECTED_NATIVE_TRANSFER_GATE - RANDOM_TRANSFER_GATE | BA | +0.0000 | [-0.0021, +0.0021] |
| OpenBMI_MI | PROTECTED_NATIVE_TRANSFER_GATE - RANDOM_TRANSFER_GATE | macro_F1 | +0.0000 | [-0.0021, +0.0022] |
| OpenBMI_MI | PROTECTED_NATIVE_TRANSFER_GATE - RANDOM_TRANSFER_GATE | worst_session_BA | +0.0007 | [+0.0000, +0.0021] |
| OpenBMI_ERP | PROTECTED_NATIVE_TRANSFER_GATE - BASELINE | BA | -0.0003 | [-0.0014, +0.0007] |
| OpenBMI_ERP | PROTECTED_NATIVE_TRANSFER_GATE - BASELINE | macro_F1 | +0.0017 | [+0.0004, +0.0031] |
| OpenBMI_ERP | PROTECTED_NATIVE_TRANSFER_GATE - BASELINE | worst_session_BA | +0.0001 | [-0.0008, +0.0011] |
| OpenBMI_ERP | PROTECTED_NATIVE_TRANSFER_GATE - P_ONLY_TRANSFER_GATE | BA | +0.0000 | [-0.0008, +0.0008] |
| OpenBMI_ERP | PROTECTED_NATIVE_TRANSFER_GATE - P_ONLY_TRANSFER_GATE | macro_F1 | +0.0006 | [-0.0002, +0.0014] |
| OpenBMI_ERP | PROTECTED_NATIVE_TRANSFER_GATE - P_ONLY_TRANSFER_GATE | worst_session_BA | -0.0003 | [-0.0010, +0.0004] |
| OpenBMI_ERP | PROTECTED_NATIVE_TRANSFER_GATE - RANDOM_TRANSFER_GATE | BA | +0.0005 | [-0.0000, +0.0011] |
| OpenBMI_ERP | PROTECTED_NATIVE_TRANSFER_GATE - RANDOM_TRANSFER_GATE | macro_F1 | +0.0008 | [+0.0002, +0.0017] |
| OpenBMI_ERP | PROTECTED_NATIVE_TRANSFER_GATE - RANDOM_TRANSFER_GATE | worst_session_BA | +0.0005 | [-0.0004, +0.0016] |
| OpenBMI_SSVEP | PROTECTED_NATIVE_TRANSFER_GATE - BASELINE | BA | +0.0000 | [-0.0021, +0.0021] |
| OpenBMI_SSVEP | PROTECTED_NATIVE_TRANSFER_GATE - BASELINE | macro_F1 | +0.0001 | [-0.0021, +0.0024] |
| OpenBMI_SSVEP | PROTECTED_NATIVE_TRANSFER_GATE - BASELINE | worst_session_BA | -0.0007 | [-0.0021, +0.0000] |
| OpenBMI_SSVEP | PROTECTED_NATIVE_TRANSFER_GATE - P_ONLY_TRANSFER_GATE | BA | +0.0000 | [-0.0021, +0.0021] |
| OpenBMI_SSVEP | PROTECTED_NATIVE_TRANSFER_GATE - P_ONLY_TRANSFER_GATE | macro_F1 | +0.0001 | [-0.0021, +0.0024] |
| OpenBMI_SSVEP | PROTECTED_NATIVE_TRANSFER_GATE - P_ONLY_TRANSFER_GATE | worst_session_BA | -0.0007 | [-0.0021, +0.0000] |
| OpenBMI_SSVEP | PROTECTED_NATIVE_TRANSFER_GATE - RANDOM_TRANSFER_GATE | BA | +0.0000 | [-0.0021, +0.0021] |
| OpenBMI_SSVEP | PROTECTED_NATIVE_TRANSFER_GATE - RANDOM_TRANSFER_GATE | macro_F1 | +0.0001 | [-0.0021, +0.0024] |
| OpenBMI_SSVEP | PROTECTED_NATIVE_TRANSFER_GATE - RANDOM_TRANSFER_GATE | worst_session_BA | -0.0007 | [-0.0021, +0.0000] |
| WBCIC_MI | PROTECTED_NATIVE_TRANSFER_GATE - BASELINE | BA | +0.0010 | [+0.0000, +0.0025] |
| WBCIC_MI | PROTECTED_NATIVE_TRANSFER_GATE - BASELINE | macro_F1 | +0.0011 | [+0.0000, +0.0027] |
| WBCIC_MI | PROTECTED_NATIVE_TRANSFER_GATE - BASELINE | worst_session_BA | +0.0005 | [-0.0015, +0.0025] |
| WBCIC_MI | PROTECTED_NATIVE_TRANSFER_GATE - P_ONLY_TRANSFER_GATE | BA | +0.0010 | [+0.0000, +0.0025] |
| WBCIC_MI | PROTECTED_NATIVE_TRANSFER_GATE - P_ONLY_TRANSFER_GATE | macro_F1 | +0.0011 | [+0.0000, +0.0027] |
| WBCIC_MI | PROTECTED_NATIVE_TRANSFER_GATE - P_ONLY_TRANSFER_GATE | worst_session_BA | +0.0015 | [+0.0000, +0.0030] |
| WBCIC_MI | PROTECTED_NATIVE_TRANSFER_GATE - RANDOM_TRANSFER_GATE | BA | +0.0005 | [+0.0000, +0.0015] |
| WBCIC_MI | PROTECTED_NATIVE_TRANSFER_GATE - RANDOM_TRANSFER_GATE | macro_F1 | +0.0005 | [+0.0000, +0.0015] |
| WBCIC_MI | PROTECTED_NATIVE_TRANSFER_GATE - RANDOM_TRANSFER_GATE | worst_session_BA | +0.0005 | [-0.0010, +0.0020] |

## Interpretation

The seed-zero primary result is **NATIVE_TRANSFER_NOT_ACTIONABLE**. No task has a
strictly positive 95% paired bootstrap lower bound for Protected minus Baseline
BA or Protected minus Random BA. A confidence interval touching zero does not
establish an improvement. Protected equals Random on OpenBMI MI and SSVEP to
the reported precision; its WBCIC advantage over Random is only 0.05
percentage points with a lower bound of zero. Protected is not demonstrably
better than P-only on the primary BA either. Thus the data do not support
Protected-specific actionability or a need for trial-specific transfer input.

| Task | Protected Macro-F1 | Protected worst-session BA | Protected NLL |
| --- | ---: | ---: | ---: |
| OpenBMI_MI | 0.7078 | 0.6814 | 0.6205 |
| OpenBMI_ERP | 0.8257 | 0.8308 | 0.2640 |
| OpenBMI_SSVEP | 0.9191 | 0.9036 | 0.3040 |
| WBCIC_MI | 0.7772 | 0.7120 | 0.5839 |

The four frozen baselines match V1 exactly across BA, Macro-F1,
worst-session BA, and NLL. The positive point estimates on MI and WBCIC are
much smaller than V1's free adapter result; this stricter axis has not shown a
useful primary benefit. These are seed-zero findings, so they do not quantify
variation across random initializations.

Gate, transfer, direction, disagreement, and rescue/harm analyses are
POST_HOC_HELDOUT_DIAGNOSTIC. Here P/C disagreement is operationalized as a
different predicted class from the full successor and its P-only
counterfactual, not as an independently trained C predictor. The gate costs
one additional frozen F0 forward per trial; see EFFICIENCY.csv.

## Reporting correction

The initial generated report assigned descriptive gain labels using a 0.005
point-estimate heuristic, including a gain label where the SSVEP difference
was zero. Those labels were removed after heldout aggregation because they
overstated the evidence. The original generated report is preserved as
FINAL_REPORT_GENERATED_INITIAL.md. This correction changes no data, code,
checkpoint, projector, selected epoch, or evaluation lock.
