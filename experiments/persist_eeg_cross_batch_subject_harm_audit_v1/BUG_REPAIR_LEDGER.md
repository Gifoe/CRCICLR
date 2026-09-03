# Bug repair ledger

1. The first dry formal start exposed an implementation-only device mismatch:
   `Delta_A` had been moved to CPU before dot products with GPU subject
   gradients. The repair keeps the displacement on the model device and only
   copies a detached CPU view for hashing. It changes no data, RNG key,
   optimizer rule, K value, batch construction, gate, or scientific claim.

2. The toy-test optimizer-state scalar was changed to use a detached tensor to
   remove a harmless autograd warning. This is diagnostic-only.

3. The norm-matched random control vector is now moved to the active model
   device before its dot product with `Delta_A`; its norm is unchanged and an
   explicit max-error check is reported.

4. Finalization exposed a validator-only fp32 roundoff issue: the largest
   stored norm-match error was `3.34e-6` after normalization and scaling in
   float32. The validator tolerance is set to `1e-5` and documented as a
   numerical tolerance. This does not alter the random direction, certificate,
   outcome, bootstrap, or any scientific decision gate.

5. The inherited validator polarity was corrected: `outcome_used=false`,
   `WBCIC_outer_opened=false`, and `OpenBMI_sealed_opened=false` are the
   required legal states, so these negative-polarity flags are checked
   explicitly rather than passed to `all()`. This only repairs reporting
   validity and cannot change a metric or gate.

The mandatory toy tests passed before the corrected formal run.
