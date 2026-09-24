# Pre-heldout protocol amendment

The original request specified five variants. Before any final-heldout array
access, the user narrowed the primary run to `PROTECTED_PC_REFINE` and the
matched canonical seed-0 `BASELINE`. A later user instruction made Random-PC a
conditional follow-up only for tasks with a positive primary final-heldout BA
point difference. This follow-up is labeled exploratory post-heldout and must
never replace the V1 primary comparison.

The initial two-variant protocol lock SHA256 was
`e92734f60513aa04138670904d4a82a358c55e9a95db618e6e9ea98e3e2e5094`.
After discovery began, code additions only expanded the pre-heldout audit,
locked evaluator, post-hoc mechanism report, and conditional exploratory
follow-up. The following training and model AST hashes were verified identical
between the first two-variant source and the amended source:

| Definition | AST SHA256 |
| --- | --- |
| `PCAdapter` | `326a7ae4b5037c1e678a861094d9cf3bc9a72904e8692f425e73273abd716dca` |
| `RefinedEEGNet` | `13ccf8e2b352511dd5ae85a1f84d0e90ac077fd9c5a2257ebb27aed01c9aeb17` |
| `projector` | `d8d5964c6858f649ae70567868a8029ef1c87ba2f19ccc302025f0117f99a1ee` |
| `context_basis` | `d680a4ad169e0f06b05a548b40bfa883e1781f02baeea90d986c95a986955c98` |
| `bases` | `50c367cc8b5b4503ac7c4d0fa5c86e1c40b26b4d971fd3c84a4d88ec9f39a386` |
| `train_adapter` | `bdb3cf831454bf9eba338ed2c9a2ab8d233edeffbb250ba7c100637479badddc` |
| `train_refit_anchor` | `b0de4a9bc33b08d7be1ed1759c249dc8c40ee86b1dc1ae6c82a0696cb2ace20c` |
| `make_variant` | `a466e7bd8af47dba066e4ff2aa52c39e818bb23fa38465539b409cb3cf5732ba` |

Before final evaluation, the first lock is preserved as
`PROTOCOL_LOCK_INITIAL.json` with its hash sidecar; the current lock is
regenerated and hashed from the amended source. No final-heldout EEG array or
label was loaded during this amendment.
