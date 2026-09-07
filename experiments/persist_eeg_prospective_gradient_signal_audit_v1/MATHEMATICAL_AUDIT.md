# Mathematical and implementation audit

The pre-run toy audit checks subject disjointness, outcome exclusion, relative displacement, quadratic first-order sign, functional-call non-overwrite, frozen BatchNorm buffers, matched dropout RNG, outer-fold cluster bootstrap, class-balanced subject losses, seed-0 enforcement, and absence of `optimizer.step()` in the primary path. The executed checks are in `results/MATH_TOY_TEST.json`.

The primary update is `theta_prime = theta - eta_epsilon * g_A`, where `eta_epsilon = epsilon * ||theta|| / (||g_A|| + 1e-12)`. Parameters and buffers are passed to `torch.func.functional_call`; the frozen model state is never mutated.
