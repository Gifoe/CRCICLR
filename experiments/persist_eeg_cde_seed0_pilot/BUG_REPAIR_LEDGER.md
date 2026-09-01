# BUG REPAIR LEDGER

No outcome-driven repair is permitted.  Scientific constants, architecture,
data roles, folds and seed are frozen in `PROTOCOL_LOCK.json`.  Any future
entry in this ledger must describe only a reproducible engineering repair
(path, device, serialization, shape, numerical stability or sampler issue),
state that it was identified independently of outcome values, and record the
validation performed after the repair.

## Pre-outcome repair

- The first launch stopped before preflight because the wrapper pointed
  `PERSIST_WBCIC_CACHE` at a non-existent path inside the canonical repository.
  It was corrected to the frozen source-only WBCIC cache path.  No outcome
  values were read, and no scientific parameter was changed.
- The first OpenBMI development fold stopped before equivalence because
  PyTorch 2.6 defaulted `torch.load` to `weights_only=True` and rejected the
  trusted canonical payload's NumPy normalizer arrays.  The loader now passes
  `weights_only=False` (with an older-version fallback), and labels are copied
  before tensor conversion.  No outcome metric was read and no scientific
  parameter was changed.
