# Method

For teacher embedding `h_T`, source-certified coordinates define a finite
joint erasure `h_T_minus_P`.  The protected functional target is

`Delta_l_P_T = CENTER(l_T - l_T_minus_P)`.

The residual target is

`l_R_T = CENTER(l_T) - Delta_l_P_T`.

PERSIST-Net has independent protected and adaptive EEGNet branches and sums
their two-class logits.  Source loss is task CE plus protected-function MSE,
residual-function MSE, and class-conditioned cross-session protected-prototype
consistency.  All teacher targets are detached and divided by one source-only
teacher logit scale; their relative magnitude is not changed.

At target time the protected branch, head, buffers, and evaluation mode are
fixed.  Only the adaptive parameter set specified by the source-selected
generic recipe is optimized on Session 1 labels.  No P/U/D quantity is
recomputed and there is no router.
