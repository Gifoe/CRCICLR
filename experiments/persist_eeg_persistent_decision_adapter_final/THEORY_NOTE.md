# Theory note

Consider `W_s,k = W_0 + U diag(a_s + t_s,k) V^T`, with zero-mean independent
transient effects and finite diagonal variance. A precision-weighted mean of
independent historical estimates has inverse precision equal to the sum of
precisions (plus the shrinkage prior), so its estimation variance is no larger
than a single-session estimate under the model assumptions. An unweighted
mean is optimal only when precisions agree; an independent per-session
estimate retains transient variance. These are proposition-level statements
under explicit assumptions, not a theorem about EEG data. The implementation
uses a local diagonal-curvature approximation to Fisher information. Empirical
source evidence did not satisfy the utility gate, so the variance argument is
not evidence of predictive transfer here.
