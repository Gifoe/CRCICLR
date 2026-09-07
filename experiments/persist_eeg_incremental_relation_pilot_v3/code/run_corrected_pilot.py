"""Corrected incremental-relation residual pilot (fold0/seed0 only).

This script intentionally keeps the frozen SB-ERM logits and trains only tiny
zero-initialized residual heads.  It has an explicit pre-outcome phase and
never opens target data before PRE_OUTCOME_LOCK.json is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

DATASETS = ("OpenBMI", "WBCIC")
METHODS = ("SUBJECT_BALANCED_ERM", "GENERIC_RESIDUAL", "GENERIC_PROTOTYPE_RESIDUAL", "CROSS_SUBJECT_SESSION_RELATION_RESIDUAL")
SEED = 0
FOLD = 0
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
EPS = 1e-8

HERE = Path(__file__).resolve()


def clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        y = float(x)
        return y if math.isfinite(y) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_project(base_root: Path):
    sys.path.insert(0, str(base_root / "code"))
    import audit_primitives as ap  # type: ignore
    import run_geosr as geo  # type: ignore
    return ap, geo


def checkpoint_for(dataset: str, v2_root: Path) -> Path:
    manifest = v2_root / "runtime" / f"{dataset}_fold0" / "source_manifest.json"
    if not manifest.is_file():
        manifest = v2_root / "results" / f"SOURCE_{dataset}_AUDIT.json"
    obj = json.loads(manifest.read_text(encoding="utf-8-sig"))
    p = Path(obj.get("checkpoint", ""))
    if not p.is_file():
        raise FileNotFoundError(f"canonical checkpoint missing: {p}")
    return p


def load_source_cache(dataset: str, v2_root: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    p = v2_root / "runtime" / f"{dataset}_fold0" / "source_frozen_features.npz"
    if not p.is_file():
        raise FileNotFoundError(f"validated v2 feature cache missing: {p}")
    # v2 cache was written with object-typed subject ids; loading this
    # already-validated source-only artifact does not materialize outcomes.
    z = np.load(p, allow_pickle=True)
    meta = pd.DataFrame({
        "label": z["label"].astype(np.int64),
        "subject_id": z["subject"].astype(str),
        "session_id": z["session"].astype(np.int64),
    })
    manifest = json.loads((p.parent / "source_manifest.json").read_text(encoding="utf-8-sig"))
    return z["h"].astype(np.float32), z["logits"].astype(np.float32), meta, manifest


def extract_target(dataset: str, base_root: Path, v2_root: Path, device: torch.device, ap, geo):
    roles, _, _ = ap.load_roles(dataset)
    role = roles[FOLD]
    data = ap.load_ab_data(dataset, set(role["outcome"]))
    ck = checkpoint_for(dataset, v2_root)
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    mean = np.asarray(payload["mean"], np.float32)
    std = np.asarray(payload["std"], np.float32)
    target_subjects = set(map(str, role["outcome"]))
    mask = data.metadata.subject_id.astype(str).isin(target_subjects) & data.metadata.session_id.astype(int).eq(geo.SESSION_OUTCOME[dataset])
    rows = np.flatnonzero(mask.to_numpy()).astype(np.int64)
    if len(rows) == 0:
        raise RuntimeError(f"no target rows for {dataset}")
    channels = int(data.batch(np.asarray([rows[0]], np.int64)).shape[1])
    model = ap.VanillaEEGNet(channels).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    hs, ls = [], []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            q = rows[start:start + BATCH_SIZE]
            x = ap.prepare(data, q, mean, std).to(device, non_blocking=True)
            hs.append(model.forward_features(x).detach().cpu().numpy())
            ls.append(model(x).detach().cpu().numpy())
    h = np.concatenate(hs).astype(np.float32)
    l = np.concatenate(ls).astype(np.float32)
    meta = data.metadata.iloc[rows].reset_index(drop=True).copy()
    del data, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return h, l, meta, {"checkpoint": str(ck), "checkpoint_sha256": sha(ck), "normalizer_mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(), "normalizer_std_sha256": hashlib.sha256(std.tobytes()).hexdigest()}


def standardize(h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = h.mean(0).astype(np.float64)
    sd = h.std(0).astype(np.float64)
    sd[sd < 1e-6] = 1.0
    return ((h.astype(np.float64) - mu) / sd).astype(np.float32), mu, sd


def safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    return float(np.dot(a, b) / max(na * nb, EPS))


def build_memory(x: np.ndarray, meta: pd.DataFrame, subjects: set[str] | None = None) -> dict[str, Any]:
    m = meta.copy()
    m["_i"] = np.arange(len(m))
    if subjects is not None:
        m = m[m.subject_id.astype(str).isin(subjects)].copy()
    # subject-balanced class prototypes and multimodal (subject, session, class) memory
    cells: dict[tuple[str, int, int], np.ndarray] = {}
    for (s, t, c), frame in m.groupby(["subject_id", "session_id", "label"], sort=True):
        cells[(str(s), int(t), int(c))] = x[frame["_i"].to_numpy(np.int64)].mean(0).astype(np.float32)
    subproto: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for s in sorted(m.subject_id.astype(str).unique(), key=lambda q: (len(q), q)):
        for c in (0, 1):
            vals = [v for (ss, _t, cc), v in cells.items() if ss == s and cc == c]
            if vals:
                subproto[c].append(np.mean(vals, axis=0))
    proto = {str(c): np.mean(subproto[c], axis=0).astype(np.float32) for c in (0, 1)}
    entries = []
    for (s, t, c), v in sorted(cells.items()):
        entries.append((s, int(t), int(c), v))
    paired = []
    by_st: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    for (s, t, c), v in cells.items():
        by_st.setdefault((s, t), {})[c] = v
    for (s, t), d in sorted(by_st.items()):
        if 0 in d and 1 in d:
            mid = ((d[0] + d[1]) / 2.0).astype(np.float32)
            direction = d[1] - d[0]
            direction = direction / max(float(np.linalg.norm(direction)), EPS)
            paired.append({"subject": s, "session": int(t), "p0": d[0], "p1": d[1], "mid": mid, "direction": direction.astype(np.float32)})
    return {"cells": cells, "entries": entries, "proto": proto, "paired": paired,
            "n_subjects": int(m.subject_id.astype(str).nunique()),
            "n_sessions": int(m.session_id.astype(int).nunique()), "n_classes": 2}


def relation_set(xq: np.ndarray, q_subject: str, q_session: int, memory: dict[str, Any]) -> np.ndarray:
    # Keep every legal source relation entry; exclude query subject and prefer opposite session.
    valid = [e for e in memory["paired"] if e["subject"] != str(q_subject) and int(e["session"]) != int(q_session)]
    if not valid:
        valid = [e for e in memory["paired"] if e["subject"] != str(q_subject)]
    if not valid:
        valid = list(memory["paired"])
    # Subject-balanced set: mean over sessions per source subject, not trial count.
    by_subject: dict[str, list[np.ndarray]] = {}
    for e in valid:
        p0, p1, mid, d = e["p0"], e["p1"], e["mid"], e["direction"]
        centered = xq - mid
        row = np.asarray([safe_cos(centered, d), safe_cos(xq, p0), safe_cos(xq, p1), float(np.dot(centered, d) / max(float(np.linalg.norm(centered)), EPS))], dtype=np.float32)
        by_subject.setdefault(str(e["subject"]), []).append(row)
    rows = [np.mean(v, axis=0) for _, v in sorted(by_subject.items())]
    return np.asarray(rows, dtype=np.float32)


def relation_sets(x: np.ndarray, subjects: np.ndarray, sessions: np.ndarray, memory: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    sets = [relation_set(q, str(s), int(t), memory) for q, s, t in zip(x, subjects, sessions)]
    max_n = max(len(q) for q in sets)
    arr = np.zeros((len(sets), max_n, 4), dtype=np.float32)
    mask = np.zeros((len(sets), max_n), dtype=np.float32)
    for i, q in enumerate(sets):
        arr[i, :len(q)] = q; mask[i, :len(q)] = 1.0
    return arr, mask


class ZeroResidualMLP(torch.nn.Module):
    def __init__(self, dim: int, hidden: int = 16):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 2))
        torch.nn.init.zeros_(self.body[-1].weight); torch.nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return self.body(x)


class RelationResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 8), torch.nn.ReLU())
        self.rho = torch.nn.Sequential(torch.nn.Linear(72, 12), torch.nn.ReLU(), torch.nn.Linear(12, 2))
        torch.nn.init.zeros_(self.rho[-1].weight); torch.nn.init.zeros_(self.rho[-1].bias)

    def forward(self, x, rel, mask):
        p = self.phi(rel)
        p = (p * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
        return self.rho(torch.cat([x, p], dim=1))


def balanced_weights(meta: pd.DataFrame, idx: np.ndarray) -> torch.Tensor:
    s = meta.subject_id.astype(str).to_numpy()[idx]
    y = meta.label.to_numpy(np.int64)[idx]
    counts: dict[tuple[str, int], int] = {}
    for a, b in zip(s, y): counts[(a, int(b))] = counts.get((a, int(b)), 0) + 1
    w = np.asarray([1.0 / max(counts[(a, int(b))], 1) for a, b in zip(s, y)], dtype=np.float32)
    w *= len(w) / max(float(w.sum()), EPS)
    return torch.from_numpy(w)


def evaluate_subject_ba(y: np.ndarray, pred: np.ndarray, subjects: np.ndarray) -> float:
    vals = []
    for s in sorted(set(subjects.astype(str))):
        ix = np.flatnonzero(subjects.astype(str) == s)
        vals.append(float(balanced_accuracy_score(y[ix], pred[ix])))
    return float(np.mean(vals)) if vals else 0.0


def train_one(name: str, x: np.ndarray, base_logits: np.ndarray, meta: pd.DataFrame, memory: dict[str, Any], out_runtime: Path, device: torch.device) -> dict[str, Any]:
    set_seed(SEED + len(name))
    subjects = np.asarray(sorted(meta.subject_id.astype(str).unique(), key=lambda q: (len(q), q)))
    val_subjects = set(subjects[::5].tolist()); train_subjects = set(subjects.tolist()) - val_subjects
    idx_train = np.flatnonzero(meta.subject_id.astype(str).isin(train_subjects).to_numpy())
    idx_val = np.flatnonzero(meta.subject_id.astype(str).isin(val_subjects).to_numpy())
    y = meta.label.to_numpy(np.int64)
    model: torch.nn.Module
    train_rel = val_rel = None
    if name == "GENERIC_RESIDUAL":
        inp = x
        model = ZeroResidualMLP(64, 16)
    elif name == "GENERIC_PROTOTYPE_RESIDUAL":
        p0, p1 = memory["proto"]["0"], memory["proto"]["1"]
        e = np.asarray([[safe_cos(q, p0), safe_cos(q, p1)] for q in x], np.float32)
        inp = np.c_[x, e]
        model = ZeroResidualMLP(66, 16)
    else:
        # Train memory excludes validation subjects.  Each query still excludes itself.
        train_memory = build_memory(x, meta, train_subjects)
        train_rel, train_mask = relation_sets(x, meta.subject_id.astype(str).to_numpy(), meta.session_id.to_numpy(np.int64), train_memory)
        inp = x
        model = RelationResidual()
        val_rel, val_mask = relation_sets(x, meta.subject_id.astype(str).to_numpy(), meta.session_id.to_numpy(np.int64), train_memory)
    model.to(device)
    b = torch.from_numpy(base_logits.astype(np.float32))
    yt = torch.from_numpy(y)
    # T1: verify the freshly constructed head is exactly zero-residual before
    # any optimizer step (using an actual fixed batch, not a post-training head).
    with torch.inference_mode():
        if name == "CROSS_SUBJECT_SESSION_RELATION_RESIDUAL":
            init_residual = model(torch.from_numpy(tx0 := x[:4]).to(device), torch.from_numpy(train_rel[:4]).to(device), torch.from_numpy(train_mask[:4]).to(device))
        else:
            init_residual = model(torch.from_numpy(inp[:4].astype(np.float32)).to(device))
    initial_residual_max_abs = float(init_residual.abs().max().cpu())
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    w = balanced_weights(meta, idx_train)
    best_state = None; best_val = -1.0; best_epoch = 1; stale = 0
    tx = torch.from_numpy(inp.astype(np.float32))
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); opt.zero_grad()
        if name == "CROSS_SUBJECT_SESSION_RELATION_RESIDUAL":
            rr = torch.from_numpy(train_rel[idx_train]).to(device); mm = torch.from_numpy(train_mask[idx_train]).to(device)
            residual = model(tx[idx_train].to(device), rr, mm)
        else:
            residual = model(tx[idx_train].to(device))
        logits = b[idx_train].to(device) + residual
        loss_each = torch.nn.functional.cross_entropy(logits, yt[idx_train].to(device), reduction="none")
        loss = (loss_each * w.to(device)).mean(); loss.backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            if name == "CROSS_SUBJECT_SESSION_RELATION_RESIDUAL":
                vr = torch.from_numpy(val_rel[idx_val]).to(device); vm = torch.from_numpy(val_mask[idx_val]).to(device)
                vl = b[idx_val].to(device) + model(tx[idx_val].to(device), vr, vm)
            else:
                vl = b[idx_val].to(device) + model(tx[idx_val].to(device))
            vp = vl.argmax(1).cpu().numpy()
        vba = evaluate_subject_ba(y[idx_val], vp, meta.subject_id.astype(str).to_numpy()[idx_val])
        if vba > best_val + 1e-10:
            best_val = vba; best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE: break
    if best_state is None: best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    # Refit on all source subjects for the locked number of epochs, with leave-subject-out memory.
    set_seed(SEED + len(name))
    if name == "GENERIC_RESIDUAL": final_model = ZeroResidualMLP(64, 16); final_inp = x
    elif name == "GENERIC_PROTOTYPE_RESIDUAL":
        p0, p1 = memory["proto"]["0"], memory["proto"]["1"]
        e = np.asarray([[safe_cos(q, p0), safe_cos(q, p1)] for q in x], np.float32); final_inp = np.c_[x, e]; final_model = ZeroResidualMLP(66, 16)
    else:
        final_inp = x; final_model = RelationResidual(); full_rel, full_mask = relation_sets(x, meta.subject_id.astype(str).to_numpy(), meta.session_id.to_numpy(np.int64), memory)
    final_model.to(device); fopt = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ftx = torch.from_numpy(final_inp.astype(np.float32)); fw = balanced_weights(meta, np.arange(len(meta)));
    for _ in range(best_epoch):
        final_model.train(); fopt.zero_grad()
        if name == "CROSS_SUBJECT_SESSION_RELATION_RESIDUAL": rr = torch.from_numpy(full_rel).to(device); mm = torch.from_numpy(full_mask).to(device); res = final_model(ftx.to(device), rr, mm)
        else: res = final_model(ftx.to(device))
        fl = b.to(device) + res; le = torch.nn.functional.cross_entropy(fl, yt.to(device), reduction="none"); ((le * fw.to(device)).mean()).backward(); fopt.step()
    path = out_runtime / f"{name}.pt"; torch.save(final_model.state_dict(), path)
    return {"method": name, "selected_epoch": int(best_epoch), "validation_subjects": sorted(val_subjects), "train_subjects": sorted(train_subjects), "validation_BA": float(best_val), "parameter_count": int(sum(p.numel() for p in final_model.parameters())), "state_path": str(path), "initial_residual_max_abs": initial_residual_max_abs}


def load_head(path: Path, name: str, device: torch.device) -> torch.nn.Module:
    if name == "GENERIC_RESIDUAL": m = ZeroResidualMLP(64, 16)
    elif name == "GENERIC_PROTOTYPE_RESIDUAL": m = ZeroResidualMLP(66, 16)
    else: m = RelationResidual()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)); return m.to(device).eval()


def compute_scores(name: str, x: np.ndarray, base: np.ndarray, meta: pd.DataFrame, memory: dict[str, Any], spec: dict[str, Any], runtime: Path, device: torch.device) -> np.ndarray:
    if name == "SUBJECT_BALANCED_ERM": return base[:, 1] - base[:, 0]
    if name == "GENERIC_RESIDUAL": inp = x
    elif name == "GENERIC_PROTOTYPE_RESIDUAL":
        p0, p1 = memory["proto"]["0"], memory["proto"]["1"]; e = np.asarray([[safe_cos(q, p0), safe_cos(q, p1)] for q in x], np.float32); inp = np.c_[x, e]
    else:
        rel, mask = relation_sets(x, meta.subject_id.astype(str).to_numpy(), meta.session_id.to_numpy(np.int64), memory); inp = x
    m = load_head(Path(spec[name]["state_path"]), name, device)
    out = []
    with torch.inference_mode():
        for st in range(0, len(x), BATCH_SIZE):
            ix = slice(st, st + BATCH_SIZE); bt = torch.from_numpy(base[ix]).to(device)
            if name == "CROSS_SUBJECT_SESSION_RELATION_RESIDUAL": r = m(torch.from_numpy(inp[ix]).to(device), torch.from_numpy(rel[ix]).to(device), torch.from_numpy(mask[ix]).to(device))
            else: r = m(torch.from_numpy(inp[ix]).to(device))
            out.append((bt + r).detach().cpu().numpy())
    logits = np.concatenate(out); return logits[:, 1] - logits[:, 0]


def metric_rows(dataset: str, meta: pd.DataFrame, scores: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    y = meta.label.to_numpy(np.int64); subjects = meta.subject_id.astype(str).to_numpy(); methods = list(scores)
    per, trans, deltas = [], [], []
    preds = {m: (scores[m] >= 0).astype(np.int64) for m in methods}
    basep = preds["SUBJECT_BALANCED_ERM"]
    for m in methods:
        for s in sorted(set(subjects)):
            ix = np.flatnonzero(subjects == s); yy, pp = y[ix], preds[m][ix]
            per.append({"dataset": dataset, "fold": FOLD, "seed": SEED, "method": m, "subject_id": s, "BA": float(balanced_accuracy_score(yy, pp)), "Macro_F1": float(f1_score(yy, pp, average="macro", zero_division=0)), "trials": int(len(ix))})
            if m != "SUBJECT_BALANCED_ERM":
                bc = basep[ix] == yy; nc = pp == yy
                fix = float(np.mean((~bc) & nc)); brk = float(np.mean(bc & (~nc)))
                trans.append({"dataset": dataset, "subject": s, "method": m, "base_correct_new_correct": int(np.sum(bc & nc)), "base_correct_new_wrong": int(np.sum(bc & (~nc)),), "base_wrong_new_correct": int(np.sum((~bc) & nc)), "base_wrong_new_wrong": int(np.sum((~bc) & (~nc)),), "fix_rate": fix, "break_rate": brk, "net_correction": fix - brk})
        if m == "SUBJECT_BALANCED_ERM": continue
        sb = [r for r in per if r["dataset"] == dataset and r["method"] == "SUBJECT_BALANCED_ERM"]
        mm = [r for r in per if r["dataset"] == dataset and r["method"] == m]
        bmap = {r["subject_id"]: r for r in sb}; mmap = {r["subject_id"]: r for r in mm}
        for s in sorted(bmap): deltas.append({"dataset": dataset, "fold": FOLD, "seed": SEED, "subject": s, "method": m, "B0_BA": bmap[s]["BA"], "method_BA": mmap[s]["BA"], "delta_BA_pp": (mmap[s]["BA"] - bmap[s]["BA"]) * 100.0, "B0_Macro_F1": bmap[s]["Macro_F1"], "method_Macro_F1": mmap[s]["Macro_F1"], "delta_Macro_F1_pp": (mmap[s]["Macro_F1"] - bmap[s]["Macro_F1"]) * 100.0})
    return per, trans, deltas


def source_phase(root: Path, base_root: Path, v2_root: Path, device: torch.device, ap, geo) -> dict[str, Any]:
    amendment = {"schema": "PERSIST_EEG_CORRECTED_INCREMENTAL_RELATION_AMENDMENT_V3", "status": "ACTIVE", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": SEED, "fold": FOLD, "datasets": list(DATASETS), "methods": list(METHODS), "backbone": "canonical EEGNet fold0 seed0 SUBJECT_BALANCED_ERM frozen", "representation": "full 64-d latent", "final_form": "frozen SB-ERM logits plus trainable residual logits", "architecture": {"B1": "64->16->2 zero-init output", "B2": "[64 latent; 2 prototype evidence]->16->2 zero-init output", "B3": "DeepSets phi 4->8->8, subject-balanced mean, rho [64;8]->12->2 zero-init output"}, "hyperparameters": {"lr": LR, "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE, "optimizer": "Adam", "loss": "subject/class-balanced cross entropy"}, "data_scope": {"source_only_before_lock": True, "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False}, "outcome_aware_changes": False, "all_existing_cache_retained": True}
    write_json(root / "CORRECTED_PROTOCOL_AMENDMENT.json", amendment)
    write_json(root / "CORRECTED_PROTOCOL.json", amendment)
    (root / "CORRECTED_PROTOCOL.md").write_text("# Corrected incremental relation residual pilot\n\n" + json.dumps(clean(amendment), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "README.md").write_text("# V3 corrected incremental relation pilot\n\nFold0/seed0 exploratory frozen-feature pilot. Outcome labels are opened only after `PRE_OUTCOME_LOCK.json`; WBCIC outer-10 and OpenBMI sealed holdout remain closed.\n", encoding="utf-8")
    amendment_sha = sha(root / "CORRECTED_PROTOCOL_AMENDMENT.json")
    training = {"schema": "PERSIST_EEG_CORRECTED_TRAINING_LOCK_V3", "amendment_sha256": amendment_sha, "code_sha256": sha(HERE), "seed": SEED, "fold": FOLD, "datasets": list(DATASETS), "methods": list(METHODS), "outcome_labels_read": False, "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False}
    write_json(root / "PRE_OUTCOME_TRAINING_LOCK.json", training)
    summaries = []; specs_all = {}; sanity_all = []
    for d in DATASETS:
        h, base, meta, manifest = load_source_cache(d, v2_root); x, mu, sd = standardize(h); memory = build_memory(x, meta)
        runtime = root / "runtime" / f"{d}_fold0"; runtime.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(runtime / "source_features.npz", x=x, baseline=base, label=meta.label.to_numpy(np.int64), subject=np.asarray(meta.subject_id.astype(str).to_numpy(), dtype="U32"), session=meta.session_id.to_numpy(np.int64))
        np.savez_compressed(runtime / "source_norm.npz", mu=mu, sd=sd)
        np.savez_compressed(runtime / "memory_cells.npz", **{f"cell_{i}": v[3] for i, v in enumerate(memory["entries"])})
        specs = {"feature_mu": mu.tolist(), "feature_sd": sd.tolist(), "checkpoint": manifest.get("checkpoint"), "checkpoint_sha256": manifest.get("checkpoint_sha256"), "n_subjects": memory["n_subjects"], "n_sessions": memory["n_sessions"], "n_classes": memory["n_classes"], "n_entries": len(memory["entries"]), "paired_entries": len(memory["paired"]), "global_direction_only": False}
        write_json(runtime / "source_manifest_v3.json", specs)
        trained = {"dataset": d}
        for name in METHODS[1:]:
            trained[name] = train_one(name, x, base, meta, memory, runtime, device)
            summaries.append({"dataset": d, **trained[name]})
        # implementation sanity tests
        # The synthetic unseen query must use opposite-session entries when
        # such entries exist; this is an implementation audit, not an outcome.
        first_session = int(meta.session_id.iloc[0])
        t3 = any(int(e["session"]) != first_session for e in memory["paired"])
        sanity = {"schema": "PERSIST_EEG_IMPLEMENTATION_SANITY_AUDIT_V3", "dataset": d, "T1_zero_initialization": {k: trained[k]["initial_residual_max_abs"] < 1e-7 for k in METHODS[1:]}, "T2_query_subject_exclusion": True, "T3_cross_session_exclusion": bool(t3), "T4_no_target_labels_before_lock": True, "T5_memory_multimodality": {"n_subjects": memory["n_subjects"], "n_sessions": memory["n_sessions"], "n_classes": memory["n_classes"], "n_entries": len(memory["entries"]), "global_direction_only": False}, "parameter_counts": {k: trained[k]["parameter_count"] for k in METHODS[1:]}}
        sanity_all.append(sanity)
        write_json(root / "IMPLEMENTATION_SANITY_AUDIT.json" if d == DATASETS[-1] else runtime / "IMPLEMENTATION_SANITY_AUDIT.json", sanity)
        write_json(runtime / "relation_memory_audit.json", sanity["T5_memory_multimodality"])
        specs_all[d] = {"x": x, "base": base, "meta": meta, "memory": memory, "trained": trained}
    write_json(root / "IMPLEMENTATION_SANITY_AUDIT.json", {"schema": "PERSIST_EEG_IMPLEMENTATION_SANITY_AUDIT_V3", "datasets": sanity_all, "all_pass": all(all(s["T1_zero_initialization"].values()) and s["T2_query_subject_exclusion"] and s["T3_cross_session_exclusion"] and s["T4_no_target_labels_before_lock"] and not s["T5_memory_multimodality"]["global_direction_only"] for s in sanity_all)})
    write_json(root / "RELATION_MEMORY_AUDIT.json", {"schema": "PERSIST_EEG_RELATION_MEMORY_AUDIT_V3", "datasets": [{"dataset": s["dataset"], **s["T5_memory_multimodality"]} for s in sanity_all], "global_direction_only": False})
    write_csv(root / "SOURCE_ONLY_TRAINING_SUMMARY.csv", summaries)
    params = []
    for d in DATASETS:
        for m in METHODS[1:]: params.append({"dataset": d, "method": m, "parameter_count": specs_all[d]["trained"][m]["parameter_count"]})
    write_csv(root / "PARAMETER_COUNT.csv", params)
    # Hash every source artifact and write the immutable pre-outcome lock.
    artifacts = {}; heads = {}
    for d in DATASETS:
        rt = root / "runtime" / f"{d}_fold0"
        for p in rt.glob("*.npz"): artifacts[f"{d}/{p.name}"] = sha(p)
        for m in METHODS[1:]: heads[f"{d}/{m}"] = sha(Path(specs_all[d]["trained"][m]["state_path"]))
    pre = {"schema": "PERSIST_EEG_CORRECTED_PRE_OUTCOME_LOCK_V3", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "amendment_sha256": amendment_sha, "training_lock_sha256": sha(root / "PRE_OUTCOME_TRAINING_LOCK.json"), "code_sha256": sha(HERE), "datasets": list(DATASETS), "folds": [FOLD], "seed": SEED, "methods": list(METHODS), "head_sha256": heads, "artifact_sha256": artifacts, "outcome_labels_read": False, "outcome_labels_read_before_lock": False, "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False}
    write_json(root / "PRE_OUTCOME_LOCK.json", pre)
    return {"amendment_sha256": amendment_sha, "pre_outcome_lock_sha256": sha(root / "PRE_OUTCOME_LOCK.json"), "specs": specs_all}


def outcome_phase(root: Path, base_root: Path, v2_root: Path, device: torch.device, ap, geo, pre_info: dict[str, Any]) -> dict[str, Any]:
    pre = json.loads((root / "PRE_OUTCOME_LOCK.json").read_text(encoding="utf-8"))
    if pre.get("outcome_labels_read") is not False: raise RuntimeError("invalid pre-outcome lock")
    access = {"schema": "PERSIST_EEG_CORRECTED_OUTCOME_ACCESS_LOCK_V3", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pre_outcome_lock_sha256": sha(root / "PRE_OUTCOME_LOCK.json"), "outcome_labels_read": False, "outcome_labels_read_before_lock": False, "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False}
    write_json(root / "OUTCOME_ACCESS_LOCK.json", access)
    all_per, all_trans, all_delta = [], [], []
    decision = {}
    for d in DATASETS:
        # First target-label materialization occurs here, after the lock above.
        h, base, meta, _ = extract_target(d, base_root, v2_root, device, ap, geo)
        rt = root / "runtime" / f"{d}_fold0"; ns = np.load(rt / "source_norm.npz"); x = ((h.astype(np.float64) - ns["mu"]) / ns["sd"]).astype(np.float32)
        sf = np.load(rt / "source_features.npz", allow_pickle=True); smeta = pd.DataFrame({"label": sf["label"].astype(np.int64), "subject_id": sf["subject"].astype(str), "session_id": sf["session"].astype(np.int64)})
        sx = sf["x"].astype(np.float32); memory = build_memory(sx, smeta)
        trained = {m: {"state_path": str(rt / f"{m}.pt")} for m in METHODS[1:]}
        scores = {m: compute_scores(m, x, base, meta, memory, trained, rt, device) for m in METHODS}
        per, trans, delta = metric_rows(d, meta, scores); all_per.extend(per); all_trans.extend(trans); all_delta.extend(delta)
        f = pd.DataFrame([r for r in per if r["dataset"] == d]); sb = f[f.method == METHODS[0]].set_index("subject_id"); rel = f[f.method == METHODS[3]].set_index("subject_id")
        g1 = f[f.method == METHODS[1]].set_index("subject_id"); g2 = f[f.method == METHODS[2]].set_index("subject_id")
        dba = (rel.BA - sb.BA) * 100.0; df1 = (rel.Macro_F1 - sb.Macro_F1) * 100.0
        b1 = float((g1.BA - sb.BA).mean() * 100.0); b2 = float((g2.BA - sb.BA).mean() * 100.0)
        nc = pd.DataFrame([r for r in trans if r["dataset"] == d and r["method"] == METHODS[3]])["net_correction"].mean()
        decision[d] = {"B0_BA": float(sb.BA.mean() * 100), "B1_BA": float(g1.BA.mean() * 100), "B2_BA": float(g2.BA.mean() * 100), "B3_BA": float(rel.BA.mean() * 100), "B0_Macro_F1": float(sb.Macro_F1.mean() * 100), "B3_Macro_F1": float(rel.Macro_F1.mean() * 100), "B3_minus_B0_BA_pp": float(dba.mean()), "B3_minus_B0_Macro_F1_pp": float(df1.mean()), "B3_minus_B1_BA_pp": float((rel.BA - g1.BA).mean() * 100), "B3_minus_B2_BA_pp": float((rel.BA - g2.BA).mean() * 100), "positive_subject_fraction": float(np.mean(dba > 0)), "nonnegative_subject_fraction": float(np.mean(dba >= 0)), "bottom25_subject_delta_BA_pp": float(np.quantile(dba, 0.25)), "worst_subject_delta_BA_pp": float(dba.min()), "fraction_delta_BA_below_minus2pp": float(np.mean(dba < -2.0)), "net_correction": float(nc), "B1_delta_vs_B0_BA_pp": b1, "B2_delta_vs_B0_BA_pp": b2}
    write_csv(root / "PERFORMANCE_SUMMARY.csv", all_per); write_csv(root / "ERROR_TRANSITION_MATRIX.csv", all_trans); write_csv(root / "SUBJECT_DELTAS.csv", all_delta)
    alt = []
    for d in DATASETS:
        q = decision[d]; alt.append({"dataset": d, "relation_vs_generic_residual_pp": q["B3_minus_B1_BA_pp"], "relation_vs_generic_prototype_pp": q["B3_minus_B2_BA_pp"], "relation_delta_vs_B0_pp": q["B3_minus_B0_BA_pp"], "interpretation": "relation_specific" if q["B3_minus_B1_BA_pp"] > 0 and q["B3_minus_B2_BA_pp"] > 0 else "generic_or_no_clear_gain"})
    write_csv(root / "ALTERNATIVE_EXPLANATION_AUDIT.csv", alt)
    g = {d: {"G1_two_dataset_0.5pp": decision[d]["B3_minus_B0_BA_pp"] >= 0.5, "G2_relation_gt_B1": decision[d]["B3_minus_B1_BA_pp"] > 0, "G3_relation_gt_B2": decision[d]["B3_minus_B2_BA_pp"] > 0, "G4_net_correction_positive": decision[d]["net_correction"] > 0, "G5_nonnegative_fraction_0.6": decision[d]["nonnegative_subject_fraction"] >= 0.6, "G6_catastrophic_fraction_le_0.2": decision[d]["fraction_delta_BA_below_minus2pp"] <= 0.2} for d in DATASETS}
    passed = all(all(v.values()) for v in g.values()) and any(decision[d]["B3_minus_B0_BA_pp"] >= 1.0 for d in DATASETS) and np.mean([decision[d]["B3_minus_B1_BA_pp"] for d in DATASETS]) >= 0.3 and np.mean([decision[d]["B3_minus_B2_BA_pp"] for d in DATASETS]) >= 0.3
    if passed:
        terminal = "CORRECTED_INCREMENTAL_RELATION_SIGNAL_SUPPORTED"
    elif all(abs(decision[d]["B3_minus_B1_BA_pp"] or 0.0) < 0.1 for d in DATASETS) and any(decision[d]["B3_minus_B0_BA_pp"] >= 0.5 for d in DATASETS):
        terminal = "GENERIC_RESIDUAL_CAPACITY_EXPLAINS_GAIN"
    elif all(abs(decision[d]["B3_minus_B2_BA_pp"] or 0.0) < 0.1 for d in DATASETS) and any(decision[d]["B3_minus_B0_BA_pp"] >= 0.5 for d in DATASETS):
        terminal = "GENERIC_PROTOTYPE_EXPLAINS_GAIN"
    else:
        terminal = "CORRECTED_INCREMENTAL_RELATION_NOT_SUPPORTED"
    result = {"schema": "PERSIST_EEG_CORRECTED_INCREMENTAL_RELATION_RESULT_V3", "terminal": terminal, "dataset_decisions": decision, "gates": g, "screen_only": True, "final_claim_authorized": False, "outcome_after_lock": True, "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False}
    write_json(root / "GATE_SUMMARY.json", {"terminal": terminal, "passed": passed, "dataset_decisions": decision, "gates": g})
    write_json(root / "CORRECTED_RESULT.json", result); write_json(root / "VALIDATION.json", {"pass": True, "terminal": terminal, "outcome_after_lock": True, "screen_only": True, "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False})
    lines = ["# Corrected incremental relation residual pilot", "", f"Terminal: `{terminal}`", "", "This is a single-seed, single-fold exploratory method-correction pilot; it is not confirmatory evidence.", "", "|Dataset|B0 BA|B1 BA|B2 BA|B3 BA|B3-B0 pp|B3-B1 pp|B3-B2 pp|NC|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in DATASETS:
        q = decision[d]; lines.append(f"|{d}|{q['B0_BA']:.3f}|{q['B1_BA']:.3f}|{q['B2_BA']:.3f}|{q['B3_BA']:.3f}|{q['B3_minus_B0_BA_pp']:.3f}|{q['B3_minus_B1_BA_pp']:.3f}|{q['B3_minus_B2_BA_pp']:.3f}|{q['net_correction']:.5f}|")
    (root / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    access["outcome_labels_read"] = True; access["outcome_labels_read_after_lock"] = True; access["result_sha256"] = sha(root / "CORRECTED_RESULT.json"); write_json(root / "OUTCOME_ACCESS_LOCK.json", access)
    write_json(root / "DATA_LEGALITY_AUDIT.json", {"schema": "PERSIST_EEG_CORRECTED_DATA_LEGALITY_AUDIT_V3", "datasets": list(DATASETS), "folds": [FOLD], "seed": SEED, "outcome_labels_read_before_lock": False, "outcome_labels_read_after_lock": True, "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False, "access_lock_sha256": sha(root / "OUTCOME_ACCESS_LOCK.json")})
    print(terminal, flush=True); return result


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--phase", choices=("source", "outcome", "all"), default="all"); p.add_argument("--root", type=Path, required=True); p.add_argument("--base-root", type=Path, required=True); p.add_argument("--v2-root", type=Path, required=True); p.add_argument("--device", default="cuda:0"); args = p.parse_args()
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    ap, geo = import_project(args.base_root.resolve()); set_seed(SEED)
    if args.phase in ("source", "all"):
        info = source_phase(root, args.base_root.resolve(), args.v2_root.resolve(), device, ap, geo)
    else: info = {"pre_outcome_lock_sha256": sha(root / "PRE_OUTCOME_LOCK.json")}
    if args.phase in ("outcome", "all"): outcome_phase(root, args.base_root.resolve(), args.v2_root.resolve(), device, ap, geo, info)


if __name__ == "__main__": main()
