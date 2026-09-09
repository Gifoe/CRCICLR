from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import modern_common as c


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def subject_ba(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    pred = logits.argmax(1)
    values = []
    for subject in c.subject_sort(np.unique(subjects.astype(str))):
        mask = subjects.astype(str) == subject
        values.append(balanced_accuracy_score(labels[mask], pred[mask]))
    return float(np.mean(values))


def eval_subset(model: torch.nn.Module, x: np.ndarray, labels: np.ndarray, subjects: np.ndarray,
                device: torch.device, batch_size: int) -> tuple[float, np.ndarray]:
    model.eval(); pieces: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size])).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(batch)
            pieces.append(output.float().cpu().numpy())
    logits = np.concatenate(pieces, axis=0)
    return subject_ba(labels, logits, subjects), logits


def model_recipe(name: str) -> tuple[int, float, float]:
    # These are fixed, model-appropriate downstream recipes.  They are not
    # selected using outer-development outcomes.
    if name == "TCFormer":
        return 96, 3e-4, 1e-3
    if name == "ST-EEGFormer-small":
        return 8, 5e-5, 5e-2
    if name == "LaBraM-base":
        return 32, 1e-4, 5e-2
    if name == "CBraMod":
        return 32, 1e-4, 5e-2
    raise KeyError(name)


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.model not in ("TCFormer", "ST-EEGFormer-small", "LaBraM-base", "CBraMod"):
        raise ValueError(args.model)
    folds, search, split_sha = c.load_split()
    fold = next(row for row in folds[args.dataset] if int(row["fold_id"]) == args.fold)
    train_subset = c.load_subset(args.dataset, fold["inner_train_subjects"], c.SOURCE_SESSIONS[args.dataset])
    val_subset = c.load_subset(args.dataset, fold["inner_val_subjects"], (c.EVAL_SESSION,))
    mean, std, norm_record = c.normalizer(args.dataset, fold["inner_train_subjects"])
    train_x = c.normalize(train_subset, mean, std)
    val_x = c.normalize(val_subset, mean, std)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c.set_seed(c.stable_seed("modern", args.model, args.dataset, args.fold, c.SEED))
    model = c.build_model(args.model, args.dataset, train_subset.channels).to(device)
    init_sha = state_hash(model)
    batch_size, lr, weight_decay = model_recipe(args.model)
    out_dir = c.modern_dir(args.dataset, args.fold, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "checkpoint_latest.pt"
    selected_path = out_dir / "selected_best.pt"
    record_path = out_dir / "record.json"
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1; history: list[dict[str, object]] = []; best = -float("inf"); best_epoch = 0; best_state = None
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        if saved.get("init_sha") != init_sha or saved.get("split_sha") != split_sha:
            raise RuntimeError(f"resume invariant mismatch: {latest_path}")
        model.load_state_dict(saved["current_state"]); optimizer.load_state_dict(saved["optimizer"])
        if saved.get("scaler") is not None: scaler.load_state_dict(saved["scaler"])
        start_epoch = int(saved["epoch"]) + 1; history = list(saved["history"]); best = float(saved["best"]); best_epoch = int(saved["best_epoch"])
        best_state = saved.get("best_state")
        print(f"[resume] {args.model} {args.dataset} fold={args.fold} epoch={start_epoch}", flush=True)
    # Minimal finite one-batch sanity check, recorded before training.
    if start_epoch == 1:
        sanity_x = torch.from_numpy(train_x[: min(batch_size, len(train_x))]).to(device)
        sanity_y = torch.from_numpy(train_subset.labels[: len(sanity_x)]).long().to(device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            sanity_loss = F.cross_entropy(model(sanity_x), sanity_y)
        if not torch.isfinite(sanity_loss): raise RuntimeError(f"non-finite sanity loss: {args.model}")
        sanity_loss.backward(); optimizer.zero_grad(set_to_none=True)
        model.eval()
        print(f"[sanity] {args.model} params={c.parameter_count(model)} loss={float(sanity_loss.detach()):.5f}", flush=True)
    began = time.perf_counter()
    rng = np.random.default_rng(c.stable_seed("order", args.model, args.dataset, args.fold, c.SEED))
    for epoch in range(start_epoch, c.MAX_EPOCHS + 1):
        model.train(); order = np.arange(len(train_x), dtype=np.int64); rng.shuffle(order); losses = []
        for begin in range(0, len(order), batch_size):
            idx = order[begin:begin + batch_size]
            xb = torch.from_numpy(np.ascontiguousarray(train_x[idx])).to(device, non_blocking=True)
            yb = torch.from_numpy(train_subset.labels[idx]).long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(xb), yb)
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss: {args.model} epoch {epoch}")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        val_ba, _ = eval_subset(model, val_x, val_subset.labels, val_subset.subjects, device, batch_size)
        selected = val_ba > best + 1e-8
        if selected:
            best = val_ba; best_epoch = epoch; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "inner_val_subject_BA": val_ba, "selected": selected})
        payload = {"epoch": epoch, "history": history, "best": best, "best_epoch": best_epoch, "best_state": best_state,
                   "current_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "init_sha": init_sha, "split_sha": split_sha}
        tmp = latest_path.with_suffix(".pt.part"); torch.save(payload, tmp); os.replace(tmp, latest_path)
        print(f"[train] {args.model} {args.dataset} fold={args.fold} epoch={epoch:02d} loss={np.mean(losses):.5f} valBA={val_ba:.5f} best={best:.5f}", flush=True)
        if epoch >= c.MIN_EPOCHS and not selected and epoch - best_epoch >= c.PATIENCE:
            print(f"[early-stop] {args.model} {args.dataset} fold={args.fold}", flush=True); break
    if best_state is None:
        raise RuntimeError(f"no selected state: {args.model} {args.dataset} fold={args.fold}")
    # Store a compact selected state and an auditable record.  Runtime files
    # stay outside the git worktree and are never committed.
    tmp = selected_path.with_suffix(".pt.part"); torch.save(best_state, tmp); os.replace(tmp, selected_path)
    record = {"model": args.model, "dataset": args.dataset, "fold": args.fold, "seed": 0,
              "parameter_count": c.parameter_count(model),
              "selected_epoch": best_epoch, "best_inner_val_BA": best, "epochs_completed": len(history),
              "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
              "train_rows": int(len(train_x)), "validation_rows": int(len(val_x)), "normalizer": norm_record,
              "split_sha256": split_sha, "init_sha256": init_sha, "selected_checkpoint": str(selected_path),
              "selected_checkpoint_sha256": c.sha256_file(selected_path), "elapsed_seconds": time.perf_counter() - began,
              "input_adapters": {"ST-EEGFormer-small": "250Hzx1000 -> linear 128Hzx512 before official tokenization",
                                 "LaBraM-base": "250Hzx1000 -> linear 200Hzx800 then 4x200 patches",
                                 "CBraMod": "exact 5x200 patch reshape preserving cache samples"}.get(args.model, "none")}
    c.write_json(record_path, record)
    print(f"[complete] {args.model} {args.dataset} fold={args.fold} best={best:.5f} epoch={best_epoch}", flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["TCFormer", "ST-EEGFormer-small", "LaBraM-base", "CBraMod"])
    parser.add_argument("--dataset", required=True, choices=list(c.DATASETS))
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
