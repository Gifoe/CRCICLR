# Pre-heldout protocol amendment

The original request specified five variants. Before any final-heldout array
access, the user narrowed the primary run to `PROTECTED_PC_REFINE` and the
matched canonical seed-0 `BASELINE`. A later user instruction made Random-PC a
conditional follow-up only for tasks with a positive primary final-heldout BA
point difference. This follow-up is labeled exploratory post-heldout and must
never replace the V1 primary comparison.

The initial two-variant protocol lock SHA256 was
`e92734f60513aa04138670904d4a82a358c55e9a95db618e6e9ea98e3e2e5094`.
After discovery began, code additions expanded the pre-heldout audit,
locked evaluator, post-hoc mechanism report, and conditional exploratory
follow-up. A subsequent source review, still before any final-heldout access,
found that the first run used ridge alpha 0.01 for the intermediate PathFit
projector. The audited pathway code fixes `RIDGE_ALPHA=1.0`. The 0.01 run
(20 discovery and 14 completed refit cells) was stopped and archived on the
server as `pc_refine_v1_runtime/invalid_ridge_0p01`; its second protocol
lock SHA256 was
`11b673ba2e2c87122e2028e148eef398b68d5c33906180f723c22f973021622b`.
No checkpoint, selected epoch, projector, PCA basis, or discovery result from
that run is eligible for the corrected V1 analysis. All cells must be rerun
with alpha 1.0 before final lock or heldout access.

The first alpha-1.0 protocol lock SHA256 was
`c82ac9aff5eb591a4cb3868040fce6fd99982b57ad33221d356f738864967c76`.
While corrected training was running, final-lock verification was strengthened
to hash and recheck the stored basis file and both refit checkpoints. This
changes no model, training, discovery, or evaluator computation. The first
alpha-1.0 lock is preserved as `PROTOCOL_LOCK_CORRECTED_INITIAL.json`;
the active lock is regenerated from the strengthened source before heldout.

The following training and model AST hashes stayed identical between the
first two-variant source and the source before this PathFit correction.
Only `projector` changes in the corrected source:

| Definition | AST SHA256 |
| --- | --- |
| `PCAdapter` | `326a7ae4b5037c1e678a861094d9cf3bc9a72904e8692f425e73273abd716dca` |
| `RefinedEEGNet` | `13ccf8e2b352511dd5ae85a1f84d0e90ac077fd9c5a2257ebb27aed01c9aeb17` |
| `context_basis` | `d680a4ad169e0f06b05a548b40bfa883e1781f02baeea90d986c95a986955c98` |
| `bases` | `50c367cc8b5b4503ac7c4d0fa5c86e1c40b26b4d971fd3c84a4d88ec9f39a386` |
| `train_adapter` | `bdb3cf831454bf9eba338ed2c9a2ab8d233edeffbb250ba7c100637479badddc` |
| `train_refit_anchor` | `b0de4a9bc33b08d7be1ed1759c249dc8c40ee86b1dc1ae6c82a0696cb2ace20c` |
| `make_variant` | `a466e7bd8af47dba066e4ff2aa52c39e818bb23fa38465539b409cb3cf5732ba` |

Before final evaluation, the first lock is preserved as
`PROTOCOL_LOCK_INITIAL.json` with its hash sidecar; the corrected current
lock is regenerated and hashed from the alpha-1.0 source. No final-heldout
EEG array or label was loaded during this amendment.
