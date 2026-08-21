# Headroom proposition

Let the fixed candidate action set be $\mathcal{A}$, legal history be $H_s$,
and future utility of action $k$ for subject $s$ be $U_s(k)$.  For any selector
$\pi(H_s) \in \mathcal{A}$,

$$
\mathbb{E}[U_s(\pi(H_s))] \leq \mathbb{E}[\max_{k\in\mathcal{A}} U_s(k)].
$$

This follows pointwise because the selected action's utility cannot exceed the
maximum utility in the same action set.  Therefore, if the deployment-level
oracle exceeds the strongest matched baseline by less than $\epsilon$, no
selector over that bank can deliver more than $\epsilon$.  The proposition is
only an upper-bound argument; it says nothing about whether the oracle action
is predictable from legal history.

V8's baseline-updated two-fold union headroom was approximately
2.056 pp on OpenBMI and 2.083 pp on WBCIC, so selector research cannot meet the requested dual +5 pp target with this action set.
