# Upstream license and fidelity audit

Audit date: 2026-08-21.

## EEG-DG

- Upstream: `https://github.com/zxchit2022/EEG-DG`
- Audited commit: `5740e8b5707ac3ee8c3053144a735fe1b59cda14`
- License: **none present** at the repository root or in the tree.
- Reuse decision: no upstream source is copied or vendored.
- Additional reproducibility issue: the committed entry point imports
  `Shallow_Inception_Network_2source` and `tSNE_4`, neither of which is present
  at the audited commit.
- Local implementation: clean-room implementation of the documented
  multi-scale feature extractor and the published marginal-distribution,
  conditional-distribution, and source-domain objectives. It is not claimed to
  be bitwise or architecture-exact reproduction.

## SCLDGN

- Upstream: `https://github.com/hongyizhi/SCLDGN`
- Audited commit: `e712f9f0636765b6fc77421b0d6a4d1eb01cf775`
- License: **none present** at the repository root or in the tree.
- Reuse decision: no upstream source is copied or vendored.
- Audited behavior: the official training path combines task CE, pairwise
  CORAL, and supervised contrastive loss over same-class mixed feature views;
  the B7 path uses nine 4-Hz filter bands from 4--40 Hz.
- Local implementation: clean-room fixed FIR filterbank and multi-kernel
  spatial-temporal encoder with the same three objective concepts. The local
  filter realization and compact 64-D penultimate layer are explicit
  deviations.

## Consequence

Both families can support a mechanism experiment, but neither result may be
described as an exact official-code replication. A failure can only reject the
tested clean-room instantiation, not every possible implementation of the
published method.

