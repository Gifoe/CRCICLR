# Action feature dictionary

The CSV is wide by action so repeated outcomes are not presented as
independent meta-samples. `family_id` determines the legal decision unit and
the applicable action menu.

## Legal feature families

- `f_router_*`: label-free task confidence, counterfactual logit response,
  and protected-geometry features computed by subject-cross-fitting.
- `f_persistence_*`, `f_rank`: persistence evidence constructed outside the
  held-out consequence role.
- `f_u_crossrun*` / `f_u_crossouterfold*`: DDA utility priors from entirely
  different runs or outer folds; repeated audit cells from the target run are
  excluded.
- `f_u_crossfold*`: WBCIC signed-utility estimates from other outcome folds only.
- `f_u_crossbackbone*`: utility prior from other backbones only, used solely
  for leave-one-backbone-out validation.
- `f_decision_*`, `f_jacobian_*`: label-free local/finite decision response.
- `f_geometry_*`, eigenvalue/angle/condition features: pre-outcome geometry.
- `f_task_BA`, `f_baseline_BA`: development-only representation competence.

## Targets and prohibited inputs

All `effect_*` columns, target labels, action predictions, realised oracle
actions, and same-cell signed utility are outcomes. They are never supplied to
a model. The sealed WBCIC outer test is absent. `OUTER_TEST_USED = false`.

## Grouping

OpenBMI router rows are grouped by subject across every fold and seed. DDA
rows are grouped by complete run. WBCIC rows are evaluated with complete fold
or backbone holdout. Families are never pooled in a single fitted model.
