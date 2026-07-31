# Method specification

For each new subject, only unlabeled context `U_s` is exposed to feature construction and adaptation. Future `V_s` is represented by a separate episode field and is accepted only by offline evaluation functions. Candidate actions are `no_tta`, `t3a`, and a CPU schema mock for `entropy_adapter`; the mock is explicitly not evidence about real EEG adaptation.

For probabilities `p`, the nested set is `C_lambda={k:p_k>=1-lambda} union {argmax p}`. The default 20-point grid spans 0.50–0.99. Future miscoverage is aggregated into independent units: non-overlapping 10-minute sleep blocks or separate MI future runs.

The versioned engineering bound `empirical-bernstein-block-v1` uses B block risks, sample variance with `ddof=1`, and

`margin = sqrt(2 var log(3/eta)/B) + 3 log(3/eta)/B`.

It returns a conservative bound of one when fewer than three blocks exist. This is the CPU engineering version; the final manuscript theorem and implementation must use the same assumptions and constants.

On each calibration subject, the residual is maximized over the complete action–lambda surface. With `m` independent calibration subjects and `k=ceil((m+1)(1-delta))`, `q` is the kth sorted subject maximum; if `k>m`, `q=1`. This single `q` is added to every target candidate before selection. No interpolated quantile is used.

Selection is deterministic: feasible candidates have bound at most alpha, then minimize average set size, maximize singleton rate, minimize action cost, maximize lambda, and finally sort by action name. Absence of a feasible candidate is `uncertified`, never a successful certificate.

## Feasibility warning

The additive term alone is `3 log(60)/B` at the default `eta=0.05`. It exceeds 0.20 whenever `B<62`, even with zero empirical risk and variance. A typical sleep future shorter than 620 minutes therefore cannot certify `alpha=0.20` under this exact engineering bound. The CPU simulation correctly exposes this as zero CSR rather than hiding it. Before a paper claim, the protocol/bound must be changed with a matching proof or the target risk level must be reconsidered.
