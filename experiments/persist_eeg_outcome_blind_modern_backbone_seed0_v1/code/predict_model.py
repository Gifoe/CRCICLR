from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

import modern_common as c


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(c.MODELS))
    parser.add_argument("--dataset", required=True, choices=list(c.DATASETS))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--stage", choices=["A", "B"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    folds, search, split_sha = c.load_split()
    fold = next(row for row in folds[args.dataset] if int(row["fold_id"]) == args.fold)
    subjects = fold["inner_val_subjects"] if args.stage == "A" else fold["outer_dev_subjects"]
    output = c.RUNTIME / "predictions" / f"stage{args.stage}_{args.dataset.lower()}_fold{args.fold}_{c.modern_slug(args.model) if args.model not in ('EEGNet','LiteBN') else args.model.lower()}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and not args.force:
        print(f"[reuse] {output}", flush=True)
        return 0
    mean, std, norm_record = c.normalizer(args.dataset, fold["inner_train_subjects"])
    subset = c.load_subset(args.dataset, subjects, (c.EVAL_SESSION,))
    x = c.normalize(subset, mean, std)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = c.load_model_for_inference(args.model, args.dataset, args.fold, device)
    logits = c.infer(model, x, device, batch_size=8 if args.model == "ST-EEGFormer-small" else c.BATCH_SIZE)
    if logits.shape != (len(subset.labels), 2) or not np.isfinite(logits).all():
        raise RuntimeError(f"invalid logits {args.model} {args.dataset} fold={args.fold}: {logits.shape}")
    tmp = output.with_suffix(".npz.part")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, logits=logits.astype(np.float32), labels=subset.labels.astype(np.int64),
                            subjects=subset.subjects.astype("U32"), sessions=subset.sessions.astype(np.int64),
                            model=np.asarray(args.model), dataset=np.asarray(args.dataset), fold=np.asarray(args.fold),
                            stage=np.asarray(args.stage), split_sha=np.asarray(split_sha),
                            normalizer_sha=np.asarray(norm_record["mean_std_sha256"]))
    os.replace(tmp, output)
    print(f"[predict] {args.stage} {args.model} {args.dataset} fold={args.fold} rows={len(logits)} -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
