# OpenBMI ERP LiteBN-XS seed1/2 stability audit

The authoritative results are in outputs_correct_xs/. They use the actual original LiteBN-XS architecture (including the channel gate), not LiteBN-X.

An earlier implementation attempt in this branch was found to call LiteBN-X while labelling its rows as XS. Its output was removed from the branch tip and is not used as evidence. The corrected run uses an independent runtime and strict LiteBN_XS state construction.

Scope: OpenBMI ERP only; seeds 1 and 2; five existing outer-development folds; canonical inner split only. Historical LiteBN baseline checkpoints were verified and reused. No final heldout/test data were accessed.
