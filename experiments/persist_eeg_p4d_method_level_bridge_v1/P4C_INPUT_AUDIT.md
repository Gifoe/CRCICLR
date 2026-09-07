# P4C Input Audit

- Exact terminal: `P4C_SAFETY_BOUNDARY_PARTIAL_SUPPORTED`.
- Validator: `PASS`.
- Low-E actionability: `LOW_E_SUPPRESSION_NOT_BENEFICIAL`.
- P4D authorization: `CONDITIONAL`.
- P4C assignment SHA-256: `0a6c4caf19937ee9024b852173f07dc8b151d13806e0e5c4635fbca77d98db30`.
- P4C source cube SHA-256: `41c5373bd73f327a652c3d155ffcf90642589f35e48ce0b2a47ee30307443ec0`.
- P4B normalization SHA-256: `dfcbcfcde0536e5c673637ab6b300377b4162e5205ba555c90f73274b1c6720f`.
- OpenBMI sealed internal holdout: untouched.
- WBCIC outer 10: untouched and not enumerated.

The partial P4C result is not upgraded. Conditional P4D authorization follows only because pooled DeltaRegime is positive, pooled U_high is negative, both S4 and S6 share those directions, and purity passes. P4E remains unauthorized because P4C is not strong.
