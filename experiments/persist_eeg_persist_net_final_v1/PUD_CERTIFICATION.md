# P/U/D certification

1. **P**: source-only, class-conditioned subject/session centroids in a
   whitened teacher embedding define the symmetric S1-S2 cross-covariance.
   Whitening retains at most 32 eigen-directions whose covariance eigenvalue is
   greater than `max(0.001 * lambda_max, 1e-8)`.
   An individual direction must exceed a 200-draw subject-session permutation
   p95 null and have cross-session correlation >=0.05.
2. **U**: erase the finite direction while retaining the frozen teacher head.
   The subject-level balanced CE harm is paired against 64 matched-energy
   random directions.  The 10,000-draw subject-bootstrap 95% lower bound of
   protected-minus-random harm must exceed zero.
3. **D**: compute exact Exp3 finite class-centered logit RMS.  The direction
   must exceed the matched-random mean (ratio >1.0).  `D_flip` is never used.

P-only, P+U, and P+D are frozen ablations.  Identity, random, and PCA bases are
source-only matched-rank controls and cannot affect the PUD set.
