#!/usr/bin/env python3
"""LiteBN-XS zero-initialized residual classifier-head canonical-inner run.

This is a deliberately small, outcome-blind diagnostic.  It loads the exact
existing LiteBN-XS checkpoint for each task/fold, freezes every XS parameter,
precomputes its deterministic embedding stream, and trains only a zero-initial
residual linear head (dW, db) on the frozen canonical inner split.  No
outer-development or final held-out cohort is enumerated by this program.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO = Path("/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK")
BASE_EXP = REPO / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1"
EXP = REPO / "experiments/persist_eeg_xs_residual_head_seed0_v1"
OUT = EXP / "outputs_single_inner"
PROTOCOL = EXP / "protocol"
RUNTIME = Path("/root/rivermind-data/xs_residual_head_seed0_runtime")
SOURCE_RUNTIME = Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime")
SEED = 0
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-4
WEIGHT_DECAY = 5e-4
BETA = 1e-3
CLIP = 5.0
TOL = 1e-10
LOGIT_TOL = 1e-6
BATCH = 512


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    torch.save(value, tmp)
    os.replace(tmp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state_hash(state: dict[str, torch.Tensor]) -> str:
    stream = io.BytesIO()
    torch.save(state, stream)
    return sha_bytes(stream.getvalue())


def load_module():
    path = BASE_EXP / "code/litebn_x.py"
    spec = importlib.util.spec_from_file_location("xs_residual_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen LiteBN-X implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SEED = SEED
    return module


def xs_path(task: str, fold: int) -> Path:
    return SOURCE_RUNTIME / "checkpoints" / task.lower() / f"fold{fold}_litebn_xs" / "selected_best.pt"


def normalizer_path(task: str, fold: int) -> Path:
    return SOURCE_RUNTIME / "normalizers" / f"{task.lower()}_fold{fold}.npz"


def source_code_hash() -> str:
    return sha_file(BASE_EXP / "code/litebn_x.py")


class ResidualHead(nn.Module):
    """Zero-initialized dW/db applied to a frozen XS embedding."""

    def __init__(self, base: nn.Module, classes: int, embedding_dim: int):
        super().__init__()
        self.base = base
        self.dW = nn.Parameter(torch.zeros((classes, embedding_dim), dtype=torch.float32))
        self.db = nn.Parameter(torch.zeros((classes,), dtype=torch.float32))

    def forward(self, embedding: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        return base_logits + F.linear(embedding, self.dW, self.db)


def metrics(labels: np.ndarray, logits: np.ndarray, mod) -> dict[str, float]:
    return mod.classification_metrics(labels, logits)


def subject_mean_metrics(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray, mod) -> dict[str, float]:
    rows = []
    for subject in sorted(set(map(str, subjects))):
        mask = subjects.astype(str) == subject
        rows.append(metrics(labels[mask], logits[mask], mod))
    return {key: float(np.mean([row[key] for row in rows])) for key in ("BA", "macro_F1", "accuracy")}


def canonical_split(mod, task: str, fold: dict[str, Any], bundle) -> dict[str, Any]:
    train = mod.subject_sort(fold["inner_train_subjects"], bundle.name)
    val = mod.subject_sort(fold["inner_val_subjects"], bundle.name)
    if set(train) & set(val) or set(train) | set(val) != set(fold["inner_train_subjects"]) | set(fold["inner_val_subjects"]):
        raise RuntimeError(f"invalid canonical split {task}/fold{fold['fold_id']}")
    if set(train) & set(fold["outer_dev_subjects"]) or set(val) & set(fold["outer_dev_subjects"]):
        raise RuntimeError(f"outer leakage in canonical split {task}/fold{fold['fold_id']}")
    mean, std, meta = mod.normalizer(bundle, train)
    return {"train_subjects": train, "val_subjects": val, "mean": mean, "std": std, "normalizer_sha256": meta["mean_std_sha256"]}


def precompute(mod, model: nn.Module, bundle, cache, split: dict[str, Any], task: str, device: torch.device) -> dict[str, Any]:
    model.eval()
    source = bundle.indices(split["train_subjects"], mod.TASKS[task]["source_sessions"])
    validation = bundle.indices(split["val_subjects"], (int(mod.TASKS[task]["future_session"]),))
    def collect(indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        zs, ls, ys, ss = [], [], [], []
        with torch.no_grad():
            for start in range(0, len(indices), 128):
                batch_idx = indices[start:start + 128]
                value, labels = cache.batch(batch_idx, split["mean"], split["std"])
                logits, z = model(value)
                zs.append(z.float().detach())
                ls.append(logits.float().detach())
                ys.append(labels.detach().cpu().numpy())
                ss.extend(bundle.rows[int(i)].subject for i in batch_idx)
        return torch.cat(zs), torch.cat(ls), np.concatenate(ys), np.asarray(ss, dtype=object)
    train_z, train_l, train_y, train_s = collect(source)
    val_z, val_l, val_y, val_s = collect(validation)
    return {"train_z": train_z, "train_l": train_l, "train_y": train_y, "train_s": train_s, "val_z": val_z, "val_l": val_l, "val_y": val_y, "val_s": val_s}


def exact_replay(mod, folds: dict[str, list[dict[str, Any]]], device: torch.device) -> pd.DataFrame:
    rows = []
    for task in mod.TASK_ORDER:
        for fold in folds[mod.TASKS[task]["dataset"]]:
            fold_id = int(fold["fold_id"])
            source = xs_path(task, fold_id)
            bundle = mod.build_bundle(task, fold["inner_train_subjects"] + fold["inner_val_subjects"])
            mean, std, _ = mod.load_tensor_pair(normalizer_path(task, fold_id))
            cache = mod.RawGPUCache(bundle, device)
            base = mod.build_model("LiteBN_XS", task).to(device)
            base.load_state_dict(torch.load(source, map_location=device, weights_only=False), strict=True)
            base.eval()
            wrapper = ResidualHead(base, int(mod.TASKS[task]["classes"]), int(base.head.in_features)).to(device)
            wrapper.eval()
            indices = bundle.indices(fold["inner_val_subjects"], (int(mod.TASKS[task]["future_session"]),))
            diffs, mismatches, labels, xs_logits, residual_logits = [], 0, [], [], []
            with torch.no_grad():
                for start in range(0, len(indices), 128):
                    idx = indices[start:start + 128]
                    value, y = cache.batch(idx, mean, std)
                    xlogit, z = base(value)
                    rlogit = wrapper(z, xlogit)
                    diffs.append(float((xlogit - rlogit).abs().max().cpu()))
                    mismatches += int((xlogit.argmax(1) != rlogit.argmax(1)).sum().cpu())
                    labels.append(y.cpu().numpy()); xs_logits.append(xlogit.cpu().numpy()); residual_logits.append(rlogit.cpu().numpy())
            y = np.concatenate(labels); xlog = np.concatenate(xs_logits); rlog = np.concatenate(residual_logits)
            xm, rm = metrics(y, xlog, mod), metrics(y, rlog, mod)
            passed = max(diffs, default=0.0) < LOGIT_TOL and mismatches == 0 and all(abs(xm[k] - rm[k]) < 1e-12 for k in xm)
            rows.append({"task": task, "fold": fold_id, "XS_checkpoint": str(source), "XS_checkpoint_sha256": sha_file(source), "normalizer_sha256": mod.load_tensor_pair(normalizer_path(task, fold_id))[2]["mean_std_sha256"], "max_abs_logit_difference": max(diffs, default=0.0), "prediction_mismatch_count": mismatches, "XS_BA": xm["BA"], "residual_epoch0_BA": rm["BA"], "XS_macro_F1": xm["macro_F1"], "residual_epoch0_macro_F1": rm["macro_F1"], "XS_accuracy": xm["accuracy"], "residual_epoch0_accuracy": rm["accuracy"], "pass": bool(passed)})
            del base, wrapper, cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(["task", "fold"]).reset_index(drop=True)
    write_csv(OUT / "EPOCH0_EXACT_XS_REPLAY.csv", frame)
    if len(frame) != 20 or not bool(frame["pass"].all()):
        raise RuntimeError("epoch0 XS exact replay failed")
    print("EPOCH0_EXACT_XS_REPLAY_PASS 20/20", flush=True)
    return frame


def cell_path(task: str, fold: int) -> Path:
    return RUNTIME / "canonical_inner" / task.lower() / f"fold{fold}"


def evaluate_cached(head: ResidualHead, arrays: dict[str, Any], split: str, mod) -> dict[str, float]:
    z, base_l, y, subjects = arrays[f"{split}_z"], arrays[f"{split}_l"], arrays[f"{split}_y"], arrays[f"{split}_s"]
    with torch.no_grad():
        logits = head(z, base_l).float().cpu().numpy()
    return subject_mean_metrics(y, logits, subjects, mod)


def train_cell(mod, task: str, fold: dict[str, Any], split: dict[str, Any], source_sha: str, device: torch.device) -> dict[str, Any]:
    fold_id = int(fold["fold_id"])
    directory = cell_path(task, fold_id)
    record_path, latest_path, best_path = directory / "record.json", directory / "latest.pt", directory / "best.pt"
    invariant = {"task": task, "fold": fold_id, "seed": SEED, "source_xs_sha256": source_sha, "normalizer_sha256": split["normalizer_sha256"], "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr": LR, "weight_decay": WEIGHT_DECAY, "beta": BETA, "batch": BATCH, "inner_split_policy": "canonical_frozen"}
    if record_path.is_file():
        value = json.loads(record_path.read_text(encoding="utf-8"))
        if value.get("invariant") == invariant:
            return value
        raise RuntimeError(f"invariant mismatch in completed cell {record_path}")
    bundle = mod.build_bundle(task, fold["inner_train_subjects"] + fold["inner_val_subjects"])
    cache = mod.RawGPUCache(bundle, device)
    base = mod.build_model("LiteBN_XS", task).to(device)
    source = xs_path(task, fold_id)
    base.load_state_dict(torch.load(source, map_location=device, weights_only=False), strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    arrays = precompute(mod, base, bundle, cache, split, task, device)
    classes, embedding_dim = int(mod.TASKS[task]["classes"]), int(arrays["train_z"].shape[1])
    head = ResidualHead(base, classes, embedding_dim).to(device)
    head.base.eval()
    for parameter in head.base.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([head.dW, head.db], lr=LR, weight_decay=WEIGHT_DECAY)
    class_weight, _ = mod.class_weights(bundle, split["train_subjects"])
    class_weight = None if class_weight is None else class_weight.to(device)
    epoch0 = evaluate_cached(head, arrays, "val", mod)
    history = [{"epoch": 0, **epoch0, "delta_BA": 0.0, "dW_norm": 0.0, "db_norm": 0.0, "loss": None, "best_epoch": 0, "bad_epochs": 0}]
    best_ba, best_epoch, bad_epochs = float(epoch0["BA"]), 0, 0
    torch_save(best_path, {"invariant": invariant, "epoch": 0, "dW": head.dW.detach().cpu(), "db": head.db.detach().cpu(), "history": history, "best_epoch": 0, "best_ba": best_ba})
    train_z, train_l = arrays["train_z"], arrays["train_l"]
    train_y = torch.as_tensor(arrays["train_y"], dtype=torch.long, device=device)
    set_seed(SEED + 10_003 + fold_id)
    for epoch in range(1, MAX_EPOCHS + 1):
        head.train(); head.base.eval()
        order = torch.randperm(train_z.shape[0], device=device)
        losses = []
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            optimizer.zero_grad(set_to_none=True)
            logits = head(train_z.index_select(0, idx), train_l.index_select(0, idx))
            ce = F.cross_entropy(logits, train_y.index_select(0, idx), weight=class_weight)
            reg = BETA * (head.dW.square().mean() + head.db.square().mean())
            loss = ce + reg
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite residual-head loss {task}/f{fold_id}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([head.dW, head.db], CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        head.eval()
        val = evaluate_cached(head, arrays, "val", mod)
        improved = bool(val["BA"] > best_ba + TOL)
        if improved:
            best_ba, best_epoch, bad_epochs = float(val["BA"]), epoch, 0
        else:
            bad_epochs += 1
        dwn = float(torch.linalg.vector_norm(head.dW.detach()).cpu())
        dbn = float(torch.linalg.vector_norm(head.db.detach()).cpu())
        snapshot = {"epoch": epoch, **val, "delta_BA": float(val["BA"] - epoch0["BA"]), "dW_norm": dwn, "db_norm": dbn, "loss": float(np.mean(losses)), "best_epoch": best_epoch, "best_ba": best_ba, "bad_epochs": bad_epochs}
        history.append(snapshot)
        state = {"invariant": invariant, "epoch": epoch, "dW": head.dW.detach().cpu(), "db": head.db.detach().cpu(), "history": history, "best_epoch": best_epoch, "best_ba": best_ba}
        torch_save(latest_path, state)
        if improved:
            torch_save(best_path, state)
        print(f"[XS_RESIDUAL_HEAD {task} f{fold_id}] epoch={epoch:02d} valBA={val['BA']:.5f} best_epoch={best_epoch:02d} bestBA={best_ba:.5f} bad={bad_epochs}/{PATIENCE}", flush=True)
        if bad_epochs >= PATIENCE:
            print(f"[XS_RESIDUAL_HEAD {task} f{fold_id}] EARLY_STOP epoch={epoch:02d} best_epoch={best_epoch:02d}", flush=True)
            break
    record = {"invariant": invariant, "task": task, "dataset": mod.TASKS[task]["dataset"], "fold": fold_id, "history": history, "best_epoch": int(best_epoch), "best_inner_val_BA": float(best_ba), "best_checkpoint": str(best_path), "source_xs_checkpoint": str(source), "source_xs_sha256": source_sha, "normalizer_sha256": split["normalizer_sha256"], "inner_split_policy": "canonical_frozen"}
    write_json(record_path, record)
    del base, head, cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def select(record: dict[str, Any]) -> dict[str, Any]:
    epoch0 = record["history"][0]
    selected_epoch = int(record["best_epoch"])
    selected = next(row for row in record["history"] if int(row["epoch"]) == selected_epoch)
    delta = float(selected["BA"] - epoch0["BA"])
    status = "positive" if delta > TOL else ("zero" if abs(delta) <= TOL else "negative")
    return {"task": record["task"], "dataset": record["dataset"], "fold": record["fold"], "epoch0_XS_BA": epoch0["BA"], "selected_BA": selected["BA"], "delta_vs_XS": delta, "delta_vs_XS_pp": 100.0 * delta, "selected_epoch": selected_epoch, "residual_head_norm": float(np.hypot(selected.get("dW_norm", 0.0), selected.get("db_norm", 0.0))), "dW_norm": float(selected.get("dW_norm", 0.0)), "db_norm": float(selected.get("db_norm", 0.0)), "status": status, "early_stop_epoch": int(record["history"][-1]["epoch"]), "selected_checkpoint": record["best_checkpoint"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("pass --run")
    OUT.mkdir(parents=True, exist_ok=True); PROTOCOL.mkdir(parents=True, exist_ok=True); RUNTIME.mkdir(parents=True, exist_ok=True)
    mod = load_module()
    _, folds, split_hash = mod.load_folds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = "\n".join([
        "# LiteBN-XS zero-init residual head — seed0 canonical-inner protocol", "",
        "Exact existing LiteBN-XS checkpoints are loaded for all four tasks and five folds. The complete XS network is frozen.",
        "Only a zero-initialized residual classifier dW/db is trained on precomputed deterministic XS embeddings.",
        "One frozen canonical inner_train_subjects / inner_val_subjects split is used per fold; no additional resplits.",
        f"max_epochs={MAX_EPOCHS}; patience={PATIENCE}; lr={LR}; weight_decay={WEIGHT_DECAY}; beta={BETA}; selection=inner-val subject-mean BA.",
        "Epoch 0 is always legal and must exactly replay XS. No outer-development, final held-out/test, ensemble, gate/head fine-tuning, or embedding adapter is run.",
        f"frozen_litebn_x_source_code_sha256={source_code_hash()}", f"frozen_split_sha256={split_hash}", "FINAL_HELDOUT_ACCESSED = NO", "",
    ])
    (PROTOCOL / "EXPERIMENT_PROTOCOL_XS_RESIDUAL_HEAD_SINGLE_CANONICAL.md").write_text(protocol, encoding="utf-8")
    replay = exact_replay(mod, folds, device)
    source_sha = {(row.task, int(row.fold)): str(row.XS_checkpoint_sha256) for row in replay.itertuples(index=False)}
    records, selections, history_rows = [], [], []
    for task in mod.TASK_ORDER:
        for fold in folds[mod.TASKS[task]["dataset"]]:
            fid = int(fold["fold_id"])
            bundle = mod.build_bundle(task, fold["inner_train_subjects"] + fold["inner_val_subjects"])
            split = canonical_split(mod, task, fold, bundle)
            rec = train_cell(mod, task, fold, split, source_sha[(task, fid)], device)
            records.append(rec); selections.append(select(rec))
            for item in rec["history"]:
                history_rows.append({"task": task, "dataset": rec["dataset"], "fold": fid, **item})
            del bundle
    if len(selections) != 20:
        raise RuntimeError("incomplete XS residual-head result")
    fold_frame = pd.DataFrame(selections).sort_values(["task", "fold"]).reset_index(drop=True)
    task_rows = []
    for task, sub in fold_frame.groupby("task", sort=True):
        deltas = sub["delta_vs_XS"].to_numpy(float)
        task_rows.append({"task": task, "dataset": sub.dataset.iloc[0], "folds": len(sub), "task_mean_delta_pp": 100.0 * float(deltas.mean()), "positive_folds": int((deltas > TOL).sum()), "zero_folds": int((np.abs(deltas) <= TOL).sum()), "negative_folds": int((deltas < -TOL).sum()), "mean_selected_head_norm": float(sub.residual_head_norm.mean())})
    task_frame = pd.DataFrame(task_rows)
    write_csv(OUT / "XS_RESIDUAL_HEAD_CANONICAL_INNER_HISTORY.csv", pd.DataFrame(history_rows))
    write_csv(OUT / "XS_RESIDUAL_HEAD_CANONICAL_INNER_FOLD_RESULTS.csv", fold_frame)
    write_csv(OUT / "XS_RESIDUAL_HEAD_CANONICAL_INNER_TASK_SUMMARY.csv", task_frame)
    summary = {"regime": "XS_RESIDUAL_HEAD", "inner_split_policy": "canonical_frozen", "folds": 20, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr": LR, "weight_decay": WEIGHT_DECAY, "beta": BETA, "openbmi_erp_mean_delta_pp": float(task_frame.loc[task_frame.task == "OpenBMI_ERP", "task_mean_delta_pp"].iloc[0]), "positive_folds": int((fold_frame.status == "positive").sum()), "zero_folds": int((fold_frame.status == "zero").sum()), "negative_folds": int((fold_frame.status == "negative").sum()), "outer_development_opened": False, "final_heldout_accessed": False, "terminal": "XS_RESIDUAL_HEAD_CANONICAL_INNER_COMPLETE"}
    write_json(OUT / "XS_RESIDUAL_HEAD_CANONICAL_INNER_METADATA.json", {"experiment": "persist_eeg_xs_residual_head_seed0_v1", "summary": summary, "final_heldout_accessed": False, "outer_development_opened": False, "terminal": summary["terminal"]})
    report = ["# LiteBN-XS residual head seed0 — canonical-inner result", "", "- Exact original LiteBN-XS checkpoint per task/fold; all XS parameters frozen.", "- Only zero-initialized residual dW/db trained on precomputed XS embeddings.", "- 4 tasks × 5 folds × 1 frozen canonical inner split; no outer/test.", "", "| Task | Mean delta vs XS (pp) | Positive | Zero | Negative |", "|---|---:|---:|---:|---:|"]
    for row in task_rows:
        report.append(f"| {row['task']} | {row['task_mean_delta_pp']:+.4f} | {row['positive_folds']}/5 | {row['zero_folds']}/5 | {row['negative_folds']}/5 |")
    report += ["", f"OpenBMI ERP mean delta vs XS: **{summary['openbmi_erp_mean_delta_pp']:+.4f} pp**", "", "Selected epochs: " + "; ".join(f"{r.task}/f{int(r.fold)}={int(r.selected_epoch)}" for r in fold_frame.itertuples()), "", "`FINAL_HELDOUT_ACCESSED = NO`", ""]
    (OUT / "XS_RESIDUAL_HEAD_CANONICAL_INNER_RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print(f"XS_RESIDUAL_HEAD_CANONICAL_INNER_COMPLETE ERP={summary['openbmi_erp_mean_delta_pp']:+.4f}pp", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
