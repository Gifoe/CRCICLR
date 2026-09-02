# PERSIST-RME SEED-0 PILOT

| Dataset | Canonical seed-0 | RME seed-0 | Delta pp | paired 95% CI | matched ERM |
|---|---:|---:|---:|---|---:|

## Source-only status
{
  "OpenBMI": {
    "min_expert_delta_BA": -0.018888888888889066,
    "mean_expert_delta_BA": -0.00950000000000007,
    "mean_pairwise_disagreement": 0.02840740740740741,
    "mean_rme_delta_BA": -0.002888888888888941,
    "mean_rme_delta_NLL": -0.000588423746096256,
    "competence_pass": true,
    "diversity_pass": true,
    "rme_BA_pass": false
  },
  "WBCIC": {
    "min_expert_delta_BA": -0.13812499999999983,
    "mean_expert_delta_BA": -0.017352233270202005,
    "mean_pairwise_disagreement": 0.08191740682688958,
    "mean_rme_delta_BA": 0.0014999999999999903,
    "mean_rme_delta_NLL": 0.009832093278951081,
    "competence_pass": false,
    "diversity_pass": true,
    "rme_BA_pass": true
  },
  "cross_dataset_nll_pass": true
}

## Controls
C7 is the fixed 0.50 anchor + 0.50 four-mode risk mean. C1 is the compute-matched four-refinement ERM ensemble. C4/C5 are recorded as diagnostics identical to C7 in this bounded pilot and are not used to replace the primary method.

## Validity
- mathematical audit: PASS (`results/MATH_TOY_TEST.json`)
- canonical checkpoint equivalence: FAIL
- sealed cohorts accessed: NO
- seed 1/2 authorized/run: NO
- recipe: {'beta_risk': 0.25, 'lambda_kd': 0.5}
- runtime seconds: 11667.3

terminal = RME_SOURCE_CONSTRUCTION_FAILED
