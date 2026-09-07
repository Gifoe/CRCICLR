# TeCh provenance and license audit

Audit date: 2026-08-18 (pre-outcome)

Upstream repository: `Levi-Ackman/TeCh`

Audited commit: `9a378cc546a5d97c871eff282148175b3c7cd75b`

Paper: Yu et al., *Decentralized Attention Fails Centralized Signals:
Rethinking Transformers for Medical Time Series*, ICLR 2026,
OpenReview `oZJFY2BQt2`.

## License finding

No repository-wide `LICENSE`, `COPYING`, or SPDX declaration was present at
the audited commit.  References to CC BY-NC in two utility files cover
third-party N-BEATS-derived utilities, not TeCh as a whole.  Therefore no
permissive license for copying TeCh source can be verified.

Consequence: CRCICLR does not vendor or copy the upstream model/layer source.
The local implementation is a clean-room, paper-level faithful
reimplementation based on the publicly described architecture and observable
interface.  It must not be described as byte-identical official code.

## Architecture facts fixed before outcomes

- The model has channel-token and temporal-token branches.
- Each branch uses centralized token aggregation/redistribution (CoTAR) blocks.
- The branch outputs are pooled and added.
- The pooled sum immediately before the final linear projector is the PERSIST
  audit representation.
- WBCIC input adaptation is exactly deterministic `[B,C,T] -> [B,T,C]`.
- Training-only jitter is permitted only in its pre-registered task config.
- `eval()` and audit contain no random augmentation or test-time augmentation.
- The same input and checkpoint must return bitwise-identical CPU embeddings
  and logits in evaluation mode; this is covered by an implementation test.

This licensing limitation does not block the backbone family because the
paper supplies enough architectural detail for an independent implementation.
