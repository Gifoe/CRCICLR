# Frozen Protected/complement coupling audit

This experiment evaluates the 20 fixed EEGNet and EEGConformer OpenBMI MI/SSVEP seed-0 folds. It reuses the frozen final Protected coordinates, checkpoints, normalizers, canonical bases, splits, and deterministic equal-rank random subsets from the four preceding PERSIST-EEG audits.

Run `code/run_coupling.py --mode lock` once, then `--mode cell --model EEGNet --task OpenBMI_MI --fold 0` for each fixed cell, and `--mode aggregate` after all cells are terminal. Cell artifacts and detailed pair records remain in `COUPLING_RUNTIME`; only the compact outputs and protocol lock belong in the repository.

The audit never trains a backbone or native classifier and never loads final-heldout subjects. Surrogates fit frozen centered logits on TRAIN only; error models fit TRAIN labels using nested biological-subject CV. Class-conditioned outer-development pairs serve retrospective 2×2 evaluation only. The original future-error prediction improperly reused those label-conditioned pairs, so its AUROC is invalid. The separately provenanced label-free correction uses frozen native predictions to choose donors within an unlabeled subject/session batch; because it was specified after seeing outer results, it is exploratory and cannot establish pre-registered criterion C. See `VALIDATION_HOLD.md` and the corrected output provenance.
