# Consequence to generalization — repaired

The B0 join bug is fixed by reading the authoritative frozen `replay_per_subject.csv`. Three seeds are averaged inside each of the 40 subjects before correlation.

| predictor → outcome | n | Pearson [95% CI] | Spearman [95% CI] | interpretation |
|---|---:|---:|---:|---|
| protected_branch_erasure_harm_BA → PUD_minus_Vanilla | 40 | 0.109 [-0.272, 0.455] | 0.103 [-0.255, 0.444] | DIRECTIONAL BUT UNCERTAIN |
| protected_D_finite → PUD_minus_Vanilla | 40 | 0.140 [-0.100, 0.355] | 0.154 [-0.164, 0.449] | DIRECTIONAL BUT UNCERTAIN |
| functional_teacher_correlation → PUD_minus_Vanilla | 40 | 0.218 [-0.205, 0.528] | 0.062 [-0.289, 0.395] | DIRECTIONAL BUT UNCERTAIN |
| R_P → PUD_minus_Vanilla | 40 | -0.029 [-0.334, 0.305] | -0.023 [-0.336, 0.298] | DIRECTIONAL BUT UNCERTAIN |
| adaptive_branch_erasure_harm_BA → PUD_minus_Vanilla | 40 | 0.159 [-0.117, 0.428] | 0.165 [-0.148, 0.454] | DIRECTIONAL BUT UNCERTAIN |
| protected_branch_erasure_harm_BA → adaptation_gain | 40 | -0.165 [-0.474, 0.130] | -0.150 [-0.459, 0.186] | DIRECTIONAL BUT UNCERTAIN |

A bootstrap interval crossing zero is reported as uncertainty, not as proof of no relationship or statistical independence.
