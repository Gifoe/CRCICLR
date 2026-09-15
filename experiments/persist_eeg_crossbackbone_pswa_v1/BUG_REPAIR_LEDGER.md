# Bug / integrity ledger

- No engineering repair changes a scientific definition.
- The corrected PEEH cell JSON stores Protected ranks and assignments, but it
  does not store the frozen representation matrices, the canonical spectral
  transform, or the 100 final random-coordinate sets.
- PSWA cannot be derived from PEEH aggregate BA values because retained-subspace
  probe performance is not identifiable from erased-subspace performance.
- The locked instruction forbids recreating those missing artifacts, so affected
  cells are explicitly marked `INCOMPLETE` and no PSWA estimate is fabricated.
- A later explicit protocol repair authorized deterministic canonical-transform
  reconstruction and frozen neural inference, while continuing to prohibit any
  PEEH rerun. The repair is implemented separately in
  `code/run_pswa_recovery.py`.
- The first scheduled recovery worker terminated after a successful cell because
  Windows PowerShell 5 promoted a harmless PyTorch stderr warning to a terminating
  `NativeCommandError` under `ErrorActionPreference=Stop`. Native calls now run
  with `Continue`, preserve their actual exit code, and require the cell JSON.
- Every cell runs in a fresh Python process with at most three retries. This is
  numerically equivalent and prevents cross-cell CUDA/NumPy process-state
  contamination; completed q caches and result JSON files are never recomputed.
