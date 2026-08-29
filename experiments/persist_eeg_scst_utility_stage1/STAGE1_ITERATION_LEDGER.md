# Stage-1 iteration ledger

## 2026-08-29 — import repair

- Problem: an unused import of the old training driver relied on an implicit
  sibling-module path and broke Stage-1 startup.
- Evidence available: import traceback only; no future utility was accessed.
- Change: removed the unused driver import and retained direct loading of the
  frozen clean-room model definition.
- Prediction: source-only smoke tests and training become executable.
- Future utility inspected: no.
- Result: OpenBMI/WBCIC cache loading and all three model forward passes passed.

## 2026-08-29 — exact EEGNeX defaults

- Problem: the draft explicitly set a 32-sample first kernel although the
  Braindecode EEGNeX default is 64.
- Evidence available: installed Braindecode constructor signature; no future
  utility was accessed.
- Change: removed the override and used the official documented defaults.
- Prediction: implementation-equivalence claim is defensible.
- Future utility inspected: no.
- Result: forward smoke test passed with a 248-dimensional representation.

