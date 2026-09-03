# Method derivation

For task-only AdamW displacement Delta and stable subject gradient gbar_s=(1/4) sum of four block gradients, h_s=gbar_s dot Delta and R(Delta)=mean_s ReLU(h_s)^2. Let q=sum_{h_s>0} h_s gbar_s; then q dot Delta=sum_{h_s>0} h_s^2. The raw correction is (q dot Delta)/(||q||^2+eps) q, capped at 0.20||Delta||, and Delta_SSPG=Delta-c. Frozen backtracking accepts the first multiplier in {1,1/2,...,1/128} with non-increasing R; q=0 gives the exact identity. B gradients are post-AdamW certificates and do not alter optimizer moments or BN buffers.
