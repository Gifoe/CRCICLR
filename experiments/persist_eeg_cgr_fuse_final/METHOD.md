# Method

Two-layer width-32 GELU MLP with dropout 0.10 emits action logits and auxiliary advantages. The constrained weights are `g=s_vote**eta`, `eta∈{0.5,1,2}`, `lambda_safe∈{0.5,1}`, and five subject-bootstrap heads with `kappa∈{0,0.5}` LCB safety. Features contain only six-run decision statistics; labels, subject IDs, fold IDs, future outcomes, and oracle fields are excluded.
