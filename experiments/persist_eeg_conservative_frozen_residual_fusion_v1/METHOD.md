# Method: CFRF-v1

For frozen carrier logits `z_A`, `z_E` and their 64-dimensional frozen penultimate features `h_A`, `h_E`, CFRF feeds L2-normalized features plus `p_A`, `p_E`, and `|p_A-p_E|` into `Linear(131,32) -> GELU -> Dropout(0.10) -> Linear(32,1)`.

The final linear layer is zero-initialized. With `delta=0.25*tanh(g)`, `alpha=0.5+delta`, and `z=(1-alpha)z_A+alpha z_E`, initialization is exactly LOGIT50 and all learned weights remain in `[0.25,0.75]`. The only trainable parameters are the fusion head or the matched one-scalar `GLOBAL-ALPHA` control.

The fixed loss is CE plus `0.10 * mean(delta^2)`, AdamW (`3e-4`, `1e-3`), gradient clip 5, and 30 epochs. CFRF's primary state is epoch-30 EMA with beta 0.99. No early stopping or outer-development selection is permitted.
