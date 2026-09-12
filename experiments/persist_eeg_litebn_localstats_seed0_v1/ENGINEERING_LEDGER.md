# Execution ledger

## Source and isolation

Reused the preceding hash-verified recovery under
`D:\nips-temp\TotalP\P1\precisebn_recovered_historical`, containing original
four-task, five-fold, three-seed LiteBN checkpoints and historical source/metadata.
Only seed0 is used here. No data download, baseline checkpoint overwrite or
termination of unrelated training was performed. New runtime:
`D:\nips-temp\TotalP\P1\localstats_seed0_runtime`.

Historical code comes from the recovered repository snapshot; imports used at
execution are hashed. The CompactLite source is explicitly compared to the
recovered forward/class definition. Independent worktree branch:
`codex/persist-eeg-litebn-localstats-seed0-v1`. Do not switch the server's live
worktree while its unrelated jobs use it.

## Exact initialization and readout insertion

Reused the previously verified `legacy_init.construct_exact` helper for original
Linux float32 uniform tensor bits and historical container-hash verification on
Windows. The expected seed0 initialization SHA is checked for each task/fold;
ERP/SSVEP preserve their historical replacement-head construction sequence.
This is initialization from the original seed, not finetuning selected weights.

Forward edits are AST-checked: capture the actual branch output after
`F.elu(sn(s(y)))`, before `F.avg_pool2d`, and replace the embedding call with
the same original linear/postprocessing plus the zero-initialized supplementary
readout. The generated forward is exported. Shared parameters, BN buffers and
dropout calls are not rewritten. No module hooks or repeated backbone forwards
are used during normal training/evaluation.

V and U use an isolated Torch RNG context, with subseed741103; U is zeroed after
construction. RNG restoration and identical post-forward RNG/BN buffer states
were checked. All20 initial-function audits passed with exactly zero observed
logit difference and prediction mismatch across eval/train and FP32/AMP tests.
A finite nonzero initial U gradient is also required. Float32 statistics prevent
half-precision second-moment overflow/cancellation; the linear projections follow
the original AMP context. This choice was frozen before training.

## Training and recovery

The recovered training functions are executed with observational epoch logging
inserted after their history append. Loss and optimizer code are unchanged.
Each cell runs in a separate subprocess so all cache memory is released at cell
completion. Resume guards include code/source hashes and historical manifest,
normalizer and initialization metadata. CPU RNG tensors loaded on CUDA are moved
back to CPU before restoration, without changing their values.

The historical ERP/SSVEP extension is NOT the MI episode protocol: it uses all
session1 training trials in deterministic batch64 permutations, with ERP class
weights. These differences are intentionally preserved. All60 epoch batch-order
hashes are exported. WBCIC source-session behavior is delegated to the historical
sampler and normalizer rather than inferred from OpenBMI conventions.

## Limitations

Parameter overhead is 6,656, about13.84%-13.93% of the original model, not zero
cost. This is a seed0 development diagnostic. Current Windows Torch/CUDA differs
from the original Linux binary environment; exact initialization and sampling do
not prove the entire historical CE trajectory would be bitwise reproduced.
Dynamic AMP successful steps can differ while attempted batches match; update
audits retain this distinction. Previously exposed heldout remains diagnostic.
No seed1/2, ablation or architecture retuning is authorized by this run.
