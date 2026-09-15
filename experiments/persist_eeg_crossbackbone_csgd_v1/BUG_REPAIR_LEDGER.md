# CSGD v1 repair ledger

- Before any successful evaluation cell, the first scheduled smoke run exposed
  an OpenBMI session-index error: cache sessions are numbered 1 and 2, while
  the initial runner requested 0 and 1.  The runner and protocol lock were
  corrected to use OpenBMI indices `(1, 2)`, displayed as `(S1, S2)`.
- The first queue script also contained an invalid PowerShell interpolation
  (`$LASTEXITCODE:`).  It was corrected to `${LASTEXITCODE}` and parser-checked.
  Native stderr is now captured in the queue log before explicit exit-code
  handling.

No checkpoint, normalizer, model state, evaluation metric, or locked CSGD
definition changed.  No model training or evaluation-label-based selection
occurred during either failure.
