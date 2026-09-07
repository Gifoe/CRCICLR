# Optimizer-state audit

Task-only A gradients are the only gradients assigned to AdamW. B certificate gradients are computed with autograd but never assigned to optimizer gradients; Adam moments and BN buffers were asserted unchanged. The final parameter state is theta_old plus the registered task or corrected displacement.
