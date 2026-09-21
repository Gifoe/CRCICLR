# EEGNet/OpenBMI_SSVEP indexed-row engineering repair

## Trigger and preserved evidence

`EEGNet/OpenBMI_SSVEP/fold0` stopped with a host `MemoryError` while allocating
the 2048-by-496000 float32 temporary created by NumPy boolean row indexing in
the pathway nested-CV solver. The original `FAIL_CLOSED` JSON and scheduled log
are retained under a timestamped archive before the stale target is removed.
This was an engineering allocation failure, not a scientific terminal result.

## Repair scope

The pathway solver now uses an internal indexed-row view. It computes the same
per-column float64 TRAIN mean and standard deviation in feature chunks and
streams only those chunks to the existing frozen GPU dual-ridge solver. It
never materializes `a[mask]` as a multi-gigabyte host array. The full device
matrices and every fixed ridge-alpha GEMM retain the same shapes as before.

No checkpoint, native head, TRAIN normalizer, canonical basis, Protected
dimension set, subject split, alpha grid, random subset, outer-development
label use, or final-heldout boundary is changed. `protocol_repair_v2` remains
the implementation revision because this is a memory-layout correction only;
the changed source hash is recorded per repaired cell.

## Verification

The data-free contract suite passed after the repair, including equality of the
shared target-wise GPU dual solver to independently solved fixed targets and
the nested pathway-bank checks. Real repaired cells remain subject to their
usual exact-decomposition and protocol-lock validation.
