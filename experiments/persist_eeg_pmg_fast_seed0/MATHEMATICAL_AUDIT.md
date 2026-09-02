# Mathematical audit

The runner tests disjoint pseudo-environments, full subject coverage as B across five folds, exclusion of outcome roles, virtual `theta_prime`, detached first-order `g_A`, detached harm baselines, non-negative ReLU harm, zero harm after uniformly improving updates, identical M0 initialization, and seed=0-only invocation. `torch.func.functional_call` receives copied buffers and parameters; model parameters are never overwritten by the virtual update. No `create_graph=True` or Hessian path is used.

The executed checks are in `results/MATH_TOY_TEST.json`.
