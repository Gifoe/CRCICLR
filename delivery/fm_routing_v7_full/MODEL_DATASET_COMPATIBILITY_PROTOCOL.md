# Model–dataset compatibility protocol

Frozen before any compatibility matrix result was generated. Freeze hash: `c82564b54b288e648783c5e152a6fa5551433ada570d6a509f74248d0246ef61`.

Candidate order: `cbramod, eegpt, bendr, brant, eeg2rep, neurogpt, biot, labram`.

Dataset order: `hmc, eegmmidb, sleepedffull, bcic2a`. Combination order: `[['hmc', 'eegmmidb'], ['hmc', 'bcic2a'], ['sleepedffull', 'eegmmidb'], ['sleepedffull', 'bcic2a']]`.

## Pairwise checks

- `A_COMP_1`: signal unit has explicit evidence.
- `A_COMP_2`: sampling-rate conversion is deterministic.
- `A_COMP_3`: every channel token matches the signal semantics.
- `A_COMP_4`: forbidden bipolar-to-referential/mismatched-bipolar mappings are absent.
- `A_COMP_5`: variable-channel use has official architectural support.
- `A_COMP_6`: subject and sample coverage are each at least 0.95.
- `A_COMP_7`: official code and checkpoint forward stably.
- `A_COMP_8`: backbone is fully frozen.
- `A_COMP_9`: checkpoint lacks supervised exposure to target task labels.
- `A_COMP_10`: adapter design is independent of task performance.

The first cross-task pair with CBraMod and at least two additional compatible model families is selected. Selection may not use labels, predictions, task performance, or protected data. Failure yields `V7_STOP_NO_ADMISSIBLE_EXPERT_POOL` and terminates before Phase B.
