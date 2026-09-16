# SIRE-EEG P3 diagnostic-design bridge

## Provenance and limits

No neural network was retrained. Full/B0 is the server's historical frozen Full checkpoint, **not** a training-protocol-matched B0. Historical code evaluated Full on the same outer-development subjects. Thus Part B is a pre-specified development replay, not a strictly prospective independent confirmation.
Part A reused the exact prior seed0 Full/B1 checkpoints, Protected union and 100 equal-rank controls. The omitted train-fitted canonical basis arrays were deterministically reconstructed and checked against the frozen session-coordinate cache. Table-12 PEEH, PSWA and full-model WSBA were replayed before new summaries.
The final-heldout 14 subjects are absent from Part B. The 40 outer-development subjects were partitioned 8 per fold and opened only after SELECTOR_SPEC and SELECTION_FREEZE were written and hashed in this new run.

## Part A: WBCIC decomposition

All WSBA values are percentages; PEEH/PSWA are percentage-point contrasts.

| Model | Protected-only | Complement-retained | Intact probe | Random-only | Random complement | PEEH | PSWA | Full decoder |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical Full | 71.47 | 53.89 | 73.18 | 55.75 | 69.99 | 18.67 | 15.72 | 73.52 |
| B1 SameScale63 | 70.95 | 52.98 | 71.93 | 52.12 | 70.07 | 19.82 | 18.83 | 71.91 |

Paired B1−Full effects (20,000 biological-subject bootstrap draws):

| Metric | Mean [95% CI], pp |
|---|---:|
| protected_only_WSBA | -0.52 [-1.76, +0.63] pp |
| complement_retained_WSBA | -0.91 [-2.27, +0.70] pp |
| intact_probe_WSBA | -1.25 [-2.67, +0.03] pp |
| full_model_WSBA | -1.61 [-3.18, -0.31] pp |
| PEEH_pp | +1.15 [-0.02, +2.45] pp |
| PSWA_pp | +3.12 [+1.13, +5.00] pp |

B1 has higher relative PSWA, but its absolute Protected-only WSBA is lower and its random-only WSBA is also lower. The higher PSWA therefore must not be described as stronger absolute Protected-only prediction.
The B1−Full complement-retained point estimate is negative, but its paired CI includes zero; this decomposition does not attribute the decoder gap to complement utility alone.
The two probe utilities are not additive, and E_P(h) retains active non-P coordinates plus residual rather than a strict raw orthogonal complement.
All four tasks, including contrary directions, are in `subspace_decomposition/TASK_SUMMARY.csv` and paired effects in `FULL_VS_B1_PAIRED_EFFECTS.csv`.

## Part B: frozen development decisions and outer-development replay

| Fold | Diagnostic choice | Validation-BA choice | Development reason |
|---:|---|---|---|
| 0 | B1_SAME_SCALE_63 | B0_FULL_EXISTING | admissible; highest complement-retained WSBA |
| 1 | B2_SCALE_COLLAPSE | B0_FULL_EXISTING | admissible; highest complement-retained WSBA |
| 2 | B2_SCALE_COLLAPSE | B0_FULL_EXISTING | admissible; highest complement-retained WSBA |
| 3 | B2_SCALE_COLLAPSE | B0_FULL_EXISTING | admissible; highest complement-retained WSBA |
| 4 | B0_FULL_EXISTING | B0_FULL_EXISTING | admissible; highest complement-retained WSBA |

| Policy | Future BA | Macro-F1 | WSBA |
|---|---:|---:|---:|
| Diagnostic-guided | 75.54% | 74.74% | 72.99% |
| Random expectation | 76.15% | 75.45% | 73.32% |
| Validation BA | 78.62% | 78.02% | 75.72% |
| Fixed Full | 78.62% | 78.02% | 75.72% |
| Oracle | 78.62% | 78.02% | 75.72% |

Diagnostic-guided paired effects versus implementable comparators (percentage points; conditional subject bootstrap):

| Comparator | Δ future BA [95% CI] | Δ Macro-F1 [95% CI] | Δ WSBA [95% CI] |
|---|---:|---:|---:|
| Random expectation | -0.61 [-1.52, +0.30] | -0.71 [-1.74, +0.34] | -0.32 [-1.10, +0.48] |
| Validation BA | -3.07 [-4.34, -1.83] | -3.28 [-4.75, -1.84] | -2.72 [-3.80, -1.65] |
| Fixed Full | -3.07 [-4.34, -1.83] | -3.28 [-4.75, -1.84] | -2.72 [-3.80, -1.65] |

## Claim assessment

- Supported: relative Protected diagnostics and full-decoder quality are distinct; the WBCIC decomposition rules out equating higher PSWA with higher absolute Protected-only utility.
- Not established: the complement-retained difference alone explains the full-decoder gap; no causal mediation is inferred.
- Operational replay: The frozen diagnostic policy does not establish a positive future-BA advantage over uniform random selection: its paired CI includes or falls below zero.
- Diagnostic and validation-BA selectors differ in at least one fold; inspect the frozen fold table and paired outer effects without retuning.
- Not established: strictly prospective architectural superiority, because the historical Full arm is non-matched and its outer-development outcomes were previously exposed.

## Manuscript-ready interpretation

In frozen SIRE-EEG ablations, relative Protected-coordinate advantages did not monotonically track full-decoder robustness. A seed0 decomposition showed that Protected-only and P-removed representation probes can move separately, but did not identify a causal decomposition of decoder performance. We therefore used positive PEEH and PSWA as development-side admissibility constraints and chose among admissible architectures by complement-retained probe robustness. After freezing this rule and five fold-level decisions, we evaluated the choices on the corresponding outer-development subjects. Because the historical Full arm was not protocol-matched and had prior outer-development exposure, these outcomes are a development replay; independent P4 confirmation with the unchanged selector remains necessary.

## Artifact index

- `protocol/SOURCE_AUDIT.md`, `SOURCE_MANIFEST.json`, `PROSPECTIVE_AUDIT.json`, `SELECTOR_SPEC.json`
- `outputs/subspace_decomposition/`: checkpoint, cell, subject, task, paired-effect, replay and final reports
- `outputs/prospective_selection/`: development diagnostics, frozen choices, outer subject results, policy summaries, paired CIs and fold sensitivity
- `runtime/`: cached embeddings and run logs (server only; not committed)
