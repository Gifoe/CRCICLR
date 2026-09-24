"""Four-task, five-fold, seed-zero PC-Refine EEGNet experiment.

The only final-heldout loader is `final_eval`; it verifies FINAL_EVAL_LOCK first.
All Protected discovery is performed on legal development subjects.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
RUNTIME = Path(os.environ.get("PC_REFINE_RUNTIME", str(REPO.parent / "pc_refine_v1_runtime"))).resolve()
ANCHORS = Path(os.environ.get("PC_REFINE_ANCHORS", str(RUNTIME / "source_anchors"))).resolve()
OUTPUT = EXP / "outputs"
PROTOCOL = EXP / "protocol"
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
VARIANTS = ("BASELINE", "PROTECTED_PC_REFINE")
FOLDS = range(5)
TRUE_WBCIC = ("sub-4", "sub-8", "sub-10", "sub-15", "sub-20", "sub-39", "sub-40", "sub-43", "sub-46", "sub-51")
OPEN_CACHE = Path(os.environ.get("FULL_OPENBMI_CACHE", "/root/rivermind-data/persist_eeg_cache/openbmi/openbmi"))
WBCIC_CACHE = Path(os.environ.get("FULL_WBCIC_CACHE", "/root/rivermind-data/persist_eeg_cache/wbcic/wbcic_epochs"))
TRUE_WBCIC_CACHE = Path(os.environ.get("TRUE_OUTER_WBCIC_CACHE", "/root/rivermind-data/persist_eeg_cache/wbcic_true_outer_v1/wbcic_epochs"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(min(int(os.environ.get("PC_REFINE_CPU_THREADS", "12")), os.cpu_count() or 1))

SEVEN_CODE = REPO / "experiments" / "persist_eeg_seven_backbone_fourtask_3seed_v1" / "code"
if str(SEVEN_CODE) not in sys.path:
    sys.path.insert(0, str(SEVEN_CODE))
os.environ.setdefault("SEVEN_REPO", str(REPO))
os.environ.setdefault("SEVEN_RUNTIME", str(RUNTIME / "seven"))
os.environ.setdefault("MODERN_REPO", str(REPO))
os.environ.setdefault("TASK_GENERALITY_REPO", str(REPO))
os.environ.setdefault("FULL_OPENBMI_CACHE", str(OPEN_CACHE))
os.environ.setdefault("FULL_WBCIC_CACHE", str(WBCIC_CACHE))
os.environ.setdefault("TRUE_OUTER_WBCIC_CACHE", str(TRUE_WBCIC_CACHE))


def module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


BENCH = module("pc_refine_benchmark", SEVEN_CODE / "benchmark_data.py")
INNER = module("pc_refine_inner", SEVEN_CODE / "tech_recipe_selection.py")
MODERN = module("pc_refine_modern", REPO / "experiments" / "persist_eeg_outcome_blind_modern_backbone_seed0_v1" / "code" / "modern_common.py")
TASKMOD = module("pc_refine_taskmod", REPO / "experiments" / "persist_eeg_openbmi_task_generality_v1" / "code" / "task_datasets.py")
MODELS = module("pc_refine_models", SEVEN_CODE / "backbone_models.py")
PEEH = module("pc_refine_peeh", REPO / "experiments" / "persist_eeg_crossbackbone_peeh_v1" / "code" / "run_crossbackbone_peeh.py")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arr_sha(*values: np.ndarray) -> str:
    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32 - 1)


def seed_all(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if DEVICE.type == "cuda": torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean(x: Any) -> Any:
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x) if np.isfinite(x) else None
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    return x


def json_write(path: Path, x: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(clean(x), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=fields); w.writeheader(); w.writerows([{k: clean(row.get(k, "")) for k in fields} for row in rows])
    os.replace(temp, path)


def torch_write(path: Path, x: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    torch.save(x, temp); os.replace(temp, path)


def role(task: str, fold: int) -> tuple[dict[str, Any], str, str, tuple[int, ...], int]:
    if task in ("OpenBMI_MI", "WBCIC_MI"):
        dataset = "WBCIC" if task == "WBCIC_MI" else "OpenBMI"
        folds, _, split = MODERN.load_split()
        return next(x for x in folds[dataset] if int(x["fold_id"]) == fold), str(split), "mi", tuple(MODERN.SOURCE_SESSIONS[dataset]), int(MODERN.EVAL_SESSION)
    name = "ERP" if task == "OpenBMI_ERP" else "SSVEP"
    _, _, ref, _ = TASKMOD.split_reference()
    spec = TASKMOD.TASKS[name]
    return next(x for x in ref["folds"] if int(x["fold_id"]) == fold), str(ref["source_sha256"]), spec["cache_name"], (int(spec["source_session"]),), int(spec["future_session"])


def rows(task: str, subjects: list[str], sessions: tuple[int, ...], cache_name: str, mapping: dict[int, int] | None = None, final: bool = False):
    if task == "WBCIC_MI":
        root = TRUE_WBCIC_CACHE if final else WBCIC_CACHE
        return INNER._wbcic_rows(root, subjects, sessions, mapping)
    return INNER._openbmi_rows(OPEN_CACHE, subjects, sessions, cache_name, mapping)


def normalize(source: np.ndarray, *other: np.ndarray):
    n = source.shape[0] * source.shape[2]
    mu = (source.sum(axis=(0, 2), dtype=np.float64) / n).astype(np.float32)
    sq = np.square(source, dtype=np.float64).sum(axis=(0, 2), dtype=np.float64)
    sd = np.sqrt(np.maximum(sq / n - mu.astype(np.float64) ** 2, 1e-12)).astype(np.float32)
    f = lambda a: ((a - mu[None, :, None]) / np.maximum(sd[None, :, None], 1e-6)).astype(np.float32)
    return f(source), [f(x) for x in other], {"mean_std_sha256": hashlib.sha256(mu.tobytes() + sd.tobytes()).hexdigest()}, mu, sd


def development(task: str, fold: int, refit: bool = False) -> dict[str, Any]:
    r, split, cache_name, src_session, future_session = role(task, fold)
    train = list(map(str, r["inner_train_subjects"]))
    val = list(map(str, r["inner_val_subjects"]))
    outer = list(map(str, r["outer_dev_subjects"]))
    if (set(train) & set(val)) or (set(train) & set(outer)) or (set(val) & set(outer)):
        raise RuntimeError("development roles overlap")
    subjects = sorted(set(train + val + outer), key=lambda x: int(x.replace("sub-", ""))) if refit else train
    source, sy, ss, mapping = rows(task, subjects, src_session, cache_name)
    future, fy, fs, _ = rows(task, subjects, (future_session,), cache_name, mapping)
    if refit:
        val_x = val_y = val_s = None
    else:
        val_x, val_y, val_s, _ = rows(task, val, (future_session,), cache_name, mapping)
    normalized, others, norm, mu, sd = normalize(source, future, *([] if val_x is None else [val_x]))
    fut = others[0]; v = None if val_x is None else others[1]
    return {"source": normalized, "sy": sy.astype(np.int64), "ss": ss.astype(str), "future": fut,
            "fy": fy.astype(np.int64), "fs": fs.astype(str), "val": v, "vy": None if val_y is None else val_y.astype(np.int64),
            "vs": None if val_s is None else val_s.astype(str), "subjects": subjects, "role": r, "split": split,
            "normalizer": norm, "mu": mu, "sd": sd, "mapping": mapping, "cache_name": cache_name,
            "source_sessions": src_session, "future_session": future_session,
            "classes": int(np.unique(sy).size), "channels": int(source.shape[1]), "samples": int(source.shape[2])}


def source_anchor(task: str, fold: int) -> tuple[dict[str, Any], Path]:
    root = ANCHORS / task.lower() / f"fold{fold}_seed0"
    rec = json.loads((root / "record.json").read_text(encoding="utf-8"))
    ck = root / "selected.pt"
    if (rec["model"], rec["task"], int(rec["fold"]), int(rec["seed"])) != ("EEGNet", task, fold, 0):
        raise RuntimeError("anchor identity mismatch")
    if sha(ck) != rec["checkpoint_sha256"]:
        raise RuntimeError("anchor checkpoint hash mismatch")
    return rec, ck


def eegnet(data: dict[str, Any]) -> nn.Module:
    return MODELS.EEGNet(data["channels"], data["samples"], data["classes"]).to(DEVICE)


def load_anchor(task: str, fold: int, data: dict[str, Any]) -> tuple[nn.Module, dict[str, Any], Path]:
    rec, ck = source_anchor(task, fold)
    if rec["normalizer"]["mean_std_sha256"] != data["normalizer"]["mean_std_sha256"] or rec["split_sha256"] != data["split"]:
        raise RuntimeError("anchor split/normalizer mismatch")
    model = eegnet(data)
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True); model.eval()
    return model, rec, ck


def activate(model: nn.Module, x: np.ndarray, batch: int = 128) -> tuple[np.ndarray, np.ndarray]:
    aa, hh = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            a = torch.from_numpy(np.ascontiguousarray(x[start:start+batch])).to(DEVICE)
            v = model.drop1(model.pool1(F.elu(model.bn2(model.spatial(model.bn1(model.temporal(a.unsqueeze(1))))))))
            tail = model.drop2(model.pool2(F.elu(model.bn3(model.point(model.depth(v))))))
            h = model.embedding(tail.flatten(1))
            aa.append(v.flatten(1).float().cpu().numpy()); hh.append(h.float().cpu().numpy())
    return np.concatenate(aa), np.concatenate(hh)


def projector(centroid_a: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Exact audited PathFit ridge/standardizer and raw-space orthonormalization.
    mean = centroid_a.mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(centroid_a.std(0, dtype=np.float64).astype(np.float32), 1e-6)
    x = torch.from_numpy(np.ascontiguousarray((centroid_a - mean) / std)).to(DEVICE)
    y = torch.from_numpy(np.ascontiguousarray(targets.astype(np.float32))).to(DEVICE)
    with torch.inference_mode():
        k = x @ x.T / max(x.shape[1], 1)
        a = torch.linalg.solve(k + 1.0 * torch.eye(len(x), device=DEVICE), y)
        b = (x.T @ a / max(x.shape[1], 1)).cpu().numpy().astype(np.float32)
    u, s, _ = np.linalg.svd(b, full_matrices=False)
    rank = max(1, int((s > max(float(s[0]) * 1e-6, 1e-8)).sum()))
    q, _ = np.linalg.qr(u[:, :rank].astype(np.float64) / std[:, None].astype(np.float64), mode="reduced")
    return q.astype(np.float32), mean


def context_basis(a: np.ndarray, q: np.ndarray, mu: np.ndarray, tag: tuple[Any, ...]) -> np.ndarray:
    z = a - mu
    c = z - (z @ q) @ q.T
    rank = min(16, len(c) - 1, c.shape[1] - q.shape[1])
    if rank < 1: raise RuntimeError("empty complement")
    pca = PCA(n_components=rank, svd_solver="randomized", random_state=stable_seed("context", *tag))
    pca.fit(c)
    u = pca.components_.T.astype(np.float64)
    u -= q.astype(np.float64) @ (q.astype(np.float64).T @ u)
    u, _ = np.linalg.qr(u, mode="reduced")
    return u.astype(np.float32)


def bases(model: nn.Module, data: dict[str, Any], task: str, fold: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cap = PEEH.capped_indices
    a = cap(data["ss"], data["sy"], data["source_sessions"][0], task, fold, "train-source")
    b = cap(data["fs"], data["fy"], data["future_session"], task, fold, "train-future")
    x = np.concatenate((data["source"][a], data["future"][b]))
    y = np.concatenate((data["sy"][a], data["fy"][b]))
    s = np.concatenate((data["ss"][a], data["fs"][b])).astype(str)
    se = np.concatenate((np.full(len(a), data["source_sessions"][0]), np.full(len(b), data["future_session"]))).astype(np.int64)
    acts, h = activate(model, x, 32)
    spec = PEEH.spectrum(h, y, s, se, task, "EEGNet", fold)
    spec["classes"] = data["classes"]
    dims, assignment = PEEH.select_protected(h, y, s, se, spec, "EEGNet", task, fold)
    record = {"protected_rank": len(dims), "protected_coordinate_hash": hashlib.sha256(json.dumps(dims).encode()).hexdigest(),
              "canonical_basis_hash": arr_sha(spec["mean"], spec["basis"], spec["scale"], spec["directions"]),
              "discovery_data_hash": arr_sha(x, y, s.astype("U"), se), "assignment": assignment,
              "rank": int(spec["rank"]), "protected_coordinates": dims}
    if not dims:
        return {}, {**record, "status": "EMPTY_PROTECTED"}
    # Audited pathway mapping uses one TRAIN subject/session/class centroid per group.
    keys = sorted(set(zip(s, se, y)), key=lambda t: (int(str(t[0]).replace("sub-", "")), int(t[1]), int(t[2])))
    ca = np.stack([acts[(s == sub) & (se == ses) & (y == lab)].mean(0) for sub, ses, lab in keys]).astype(np.float32)
    ch = np.stack([h[(s == sub) & (se == ses) & (y == lab)].mean(0) for sub, ses, lab in keys]).astype(np.float32)
    targets = PEEH.canonical(ch, spec)[:, dims].astype(np.float32)
    qp, mu = projector(ca, targets)
    rd = np.sort(np.random.default_rng(PEEH.stable_seed("native-equal-rank-random", "EEGNet", task, fold, 0)).choice(spec["rank"], len(dims), replace=False)).astype(int)
    qr, mur = projector(ca, PEEH.canonical(ch, spec)[:, rd].astype(np.float32))
    if not np.array_equal(mu, mur): raise RuntimeError("random/P mapping centering disagrees")
    up = context_basis(acts, qp, mu, (task, fold, "protected"))
    ur = context_basis(acts, qr, mu, (task, fold, "random"))
    recon = acts[:32] - mu; part = (recon @ qp) @ qp.T; error = float(np.max(np.abs(recon - part - (recon - part))))
    if error >= 1e-5: raise RuntimeError(f"P/C reconstruction error {error}")
    result = {"qp": qp, "qr": qr, "mu": mu, "up": up, "ur": ur, "spec_mean": spec["mean"],
              "spec_basis": spec["basis"], "spec_scale": spec["scale"], "spec_directions": spec["directions"],
              "dims": np.asarray(dims, dtype=np.int64), "random_dims": rd}
    record.update({"status": "COMPLETE", "projector_hash": arr_sha(qp, mu), "random_subspace_hash": arr_sha(qr, mu),
                   "PCA_hash": arr_sha(up), "random_PCA_hash": arr_sha(ur), "reconstruction_error": error,
                   "projector_rank": int(qp.shape[1]), "PCA_dim": int(up.shape[1]), "random_coordinates": rd.tolist()})
    return result, record


class PCAdapter(nn.Module):
    def __init__(self, q: np.ndarray, u: np.ndarray, mu: np.ndarray):
        super().__init__()
        self.register_buffer("q", torch.from_numpy(q.copy()))
        self.register_buffer("u", torch.from_numpy(u.copy()))
        self.register_buffer("mu", torch.from_numpy(mu.copy()))
        r, c = q.shape[1], u.shape[1]
        self.wp = nn.Linear(r, 8); self.wc = nn.Linear(c, 8); self.wo = nn.Linear(8, r)
        nn.init.zeros_(self.wo.weight); nn.init.zeros_(self.wo.bias)
        self.last_delta = None; self.last_p = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        shape = h.shape; z = h.flatten(1) - self.mu
        p = z @ self.q; c = (z - p @ self.q.T) @ self.u
        delta = self.wo(F.gelu(self.wp(p)) * F.gelu(self.wc(c)))
        self.last_delta, self.last_p = delta, p
        return (h.flatten(1) + delta @ self.q.T).reshape(shape)


class POnlyAdapter(nn.Module):
    def __init__(self, q: np.ndarray, mu: np.ndarray, target_params: int):
        super().__init__()
        self.register_buffer("q", torch.from_numpy(q.copy())); self.register_buffer("mu", torch.from_numpy(mu.copy()))
        r = q.shape[1]; hidden = max(1, round((target_params - r) / (2*r + 1)))
        self.wp = nn.Linear(r, hidden); self.wo = nn.Linear(hidden, r)
        nn.init.zeros_(self.wo.weight); nn.init.zeros_(self.wo.bias)
        self.last_delta = None; self.last_p = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        shape = h.shape; p = (h.flatten(1) - self.mu) @ self.q
        delta = self.wo(F.gelu(self.wp(p)))
        self.last_delta, self.last_p = delta, p
        return (h.flatten(1) + delta @ self.q.T).reshape(shape)


class GenericAdapter(nn.Module):
    def __init__(self, channels: int, target_params: int):
        super().__init__()
        hidden = min(range(1, 128), key=lambda k: abs((2 * channels + 1) * k + channels - target_params))
        self.a1 = nn.Conv2d(channels, hidden, 1); self.a2 = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.a2.weight); nn.init.zeros_(self.a2.bias)
        self.last_delta = None; self.last_p = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta = self.a2(F.gelu(self.a1(h)))
        self.last_delta = delta.flatten(1); self.last_p = h.flatten(1)
        return h + delta


class RefinedEEGNet(nn.Module):
    def __init__(self, base: nn.Module, adapter: nn.Module):
        super().__init__(); self.base = base; self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.base; v = x.unsqueeze(1)
        v = m.drop1(m.pool1(F.elu(m.bn2(m.spatial(m.bn1(m.temporal(v)))))))
        v = self.adapter(v)
        v = m.drop2(m.pool2(F.elu(m.bn3(m.point(m.depth(v))))))
        return m.head(m.embedding(v.flatten(1)))


def make_variant(base: nn.Module, basis: dict[str, np.ndarray], variant: str) -> RefinedEEGNet:
    if variant == "PROTECTED_PC_REFINE": adapter = PCAdapter(basis["qp"], basis["up"], basis["mu"])
    elif variant == "RANDOM_PC_REFINE": adapter = PCAdapter(basis["qr"], basis["ur"], basis["mu"])
    else:
        target = sum(p.numel() for p in PCAdapter(basis["qp"], basis["up"], basis["mu"]).parameters())
        if variant == "P_ONLY_REFINE": adapter = POnlyAdapter(basis["qp"], basis["mu"], target)
        elif variant == "GENERIC_RESIDUAL": adapter = GenericAdapter(16, target)
        else: raise KeyError(variant)
        count = sum(p.numel() for p in adapter.parameters())
        if abs(count / target - 1) > 0.10: raise RuntimeError(f"adapter parameter matching failed {variant}: {count}/{target}")
    return RefinedEEGNet(copy.deepcopy(base), adapter.to(DEVICE)).to(DEVICE)


def identity_audit(base: nn.Module, model: RefinedEEGNet, x: np.ndarray) -> dict[str, Any]:
    base.eval(); model.eval()
    before = {n: (m.running_mean.detach().clone(), m.running_var.detach().clone(), m.num_batches_tracked.detach().clone())
              for n, m in model.base.named_modules() if isinstance(m, nn.BatchNorm2d)}
    with torch.inference_mode():
        batch = torch.from_numpy(np.ascontiguousarray(x[:min(8, len(x))])).to(DEVICE)
        z0, z1 = base(batch), model(batch)
    diff = float((z0 - z1).abs().max().cpu())
    equal = bool(torch.equal(z0.argmax(1), z1.argmax(1)))
    bn_equal = all(torch.equal(before[n][0], m.running_mean) and torch.equal(before[n][1], m.running_var) and torch.equal(before[n][2], m.num_batches_tracked)
                   for n, m in model.base.named_modules() if isinstance(m, nn.BatchNorm2d))
    if diff >= 1e-6 or not equal or not bn_equal: raise RuntimeError("FAIL_CLOSED zero-init identity")
    return {"logits_max_abs_diff": diff, "predictions_exact_match": equal, "BN_state_exact_match": bn_equal}


def batch_order(n: int, task: str, fold: int, phase: int, epoch: int, batch: int = 128) -> list[np.ndarray]:
    rng = np.random.default_rng(stable_seed("PC_REFINE_V1_BATCH", task, fold, phase, epoch))
    indices = rng.permutation(n)
    return [indices[i:i+batch] for i in range(0, n, batch)]


def logits(model: nn.Module, x: np.ndarray, batch: int = 128) -> np.ndarray:
    model.eval(); out = []
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            out.append(model(torch.from_numpy(np.ascontiguousarray(x[i:i+batch])).to(DEVICE)).float().cpu().numpy())
    return np.concatenate(out)


def infer_mechanism(model: nn.Module, x: np.ndarray, basis: dict[str, np.ndarray],
                    variant: str, batch: int = 128) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    head = model.head if variant == "BASELINE" else model.base.head
    spec = {"mean": basis["spec_mean"], "basis": basis["spec_basis"],
            "scale": basis["spec_scale"], "directions": basis["spec_directions"]}
    raw_base = PEEH._erasure_base(spec)
    dims = basis["dims"].astype(int)
    found: list[torch.Tensor] = []
    hook = head.register_forward_pre_hook(lambda _m, args: found.append(args[0].detach()))
    zparts: list[np.ndarray] = []
    quantities: dict[str, list[np.ndarray]] = {k: [] for k in ("correction_norm", "relative_correction", "delta_sq", "P_logit_rms", "C_logit_rms", "PC_disagreement")}
    try:
        with torch.inference_mode():
            for start in range(0, len(x), batch):
                found.clear()
                xb = torch.from_numpy(np.ascontiguousarray(x[start:start+batch])).to(DEVICE)
                z = model(xb).float()
                if len(found) != 1: raise RuntimeError("classifier input hook failed")
                h = found[0].flatten(1).float().cpu().numpy()
                q = PEEH.canonical(h, spec).astype(np.float32)
                erased = h - q[:, dims] @ raw_base[dims]
                zc = head(torch.from_numpy(np.ascontiguousarray(erased)).to(DEVICE)).float()
                zp = z - zc
                zp = zp - zp.mean(1, keepdim=True)
                zc = zc - zc.mean(1, keepdim=True)
                quantities["P_logit_rms"].append(zp.square().mean(1).sqrt().cpu().numpy())
                quantities["C_logit_rms"].append(zc.square().mean(1).sqrt().cpu().numpy())
                quantities["PC_disagreement"].append((zp.argmax(1) != zc.argmax(1)).float().cpu().numpy())
                if variant == "PROTECTED_PC_REFINE":
                    delta = model.adapter.last_delta.detach().float()
                    p = model.adapter.last_p.detach().float()
                    quantities["correction_norm"].append(delta.norm(dim=1).cpu().numpy())
                    quantities["relative_correction"].append((delta.norm(dim=1) / (p.norm(dim=1) + 1e-6)).cpu().numpy())
                    quantities["delta_sq"].append(delta.square().cpu().numpy())
                zparts.append(z.cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(zparts), {k: np.concatenate(v) for k,v in quantities.items() if v}


def score(z: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    pred = p.argmax(1)
    b, f, n = [], [], []
    for s in sorted(set(subjects.astype(str))):
        ix = subjects.astype(str) == s
        b.append(float(balanced_accuracy_score(y[ix], pred[ix])))
        f.append(float(f1_score(y[ix], pred[ix], average="macro", zero_division=0)))
        n.append(float(-np.log(np.clip(p[ix, y[ix]], 1e-12, 1)).mean()))
    return {"BA": float(np.mean(b)), "macro_F1": float(np.mean(f)), "NLL": float(np.mean(n))}


def train_adapter(model: RefinedEEGNet, data: dict[str, Any], task: str, fold: int, phase: int,
                  epochs: int, select: bool, variant: str) -> tuple[RefinedEEGNet, int, list[dict[str, Any]]]:
    seed_all(stable_seed("PC_REFINE", task, fold, phase))
    for p in model.base.parameters(): p.requires_grad_(False)
    for p in model.adapter.parameters(): p.requires_grad_(True)
    if phase == 2:
        for part in (model.base.depth, model.base.point, model.base.embedding, model.base.head):
            for p in part.parameters(): p.requires_grad_(True)
    groups = [{"params": list(model.adapter.parameters()), "lr": 3e-4 if phase == 1 else 1e-4}]
    if phase == 2:
        suffix = [p for p in model.base.parameters() if p.requires_grad]
        groups.append({"params": suffix, "lr": 1e-5})
    opt = torch.optim.AdamW(groups, weight_decay=5e-4)
    x = torch.from_numpy(np.ascontiguousarray(data["source"])).to(DEVICE)
    y = torch.as_tensor(data["sy"], dtype=torch.long, device=DEVICE)
    weight = None
    if task == "OpenBMI_ERP":
        counts = np.bincount(data["sy"], minlength=data["classes"])
        weight = torch.as_tensor(len(data["sy"]) / (data["classes"] * counts), dtype=torch.float32, device=DEVICE)
    history, best_key, best_state, best_epoch = [], None, None, 0
    for epoch in range(1, epochs + 1):
        model.eval(); model.adapter.train()
        if phase == 2:
            model.base.drop2.train()
        losses = []
        for indices in batch_order(len(y), task, fold, phase, epoch):
            idx = torch.as_tensor(indices, dtype=torch.long, device=DEVICE)
            opt.zero_grad(set_to_none=True)
            z = model(x.index_select(0, idx))
            loss = F.cross_entropy(z, y.index_select(0, idx), weight=weight)
            if variant in ("PROTECTED_PC_REFINE", "RANDOM_PC_REFINE", "P_ONLY_REFINE"):
                delta, p = model.adapter.last_delta, model.adapter.last_p
                loss = loss + 1e-4 * (delta.square().sum(1) / (p.square().sum(1) + 1e-6)).mean()
            else:
                delta, h = model.adapter.last_delta, model.adapter.last_p
                loss = loss + 1e-4 * (delta.square().sum(1) / (h.square().sum(1) + 1e-6)).mean()
            if not torch.isfinite(loss): raise RuntimeError("non-finite adapter loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step(); losses.append(float(loss.detach().cpu()))
        # BN buffers must never change in either phase.
        if select:
            zval = logits(model, data["val"])
            val = score(zval, data["vy"], data["vs"])
            key = (-val["BA"], val["NLL"], epoch)
            if best_key is None or key < best_key:
                best_key, best_epoch = key, epoch
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            val = {}
        history.append({"epoch": epoch, "phase": phase, "loss": float(np.mean(losses)), **val})
        print("ADAPTER_EPOCH", task, fold, variant, phase, epoch, history[-1], flush=True)
    if select:
        model.load_state_dict(best_state, strict=True)
    else:
        best_epoch = epochs
    return model, best_epoch, history


def cell_dir(task: str, fold: int, stage: str) -> Path:
    return RUNTIME / stage / task.lower() / f"fold{fold}_seed0"


def protocol_lock() -> None:
    path = PROTOCOL / "PROTOCOL_LOCK.json"
    if path.exists(): raise RuntimeError("protocol lock already exists")
    refs = [REPO / "experiments" / name for name in ("persist_eeg_canonical_eegnet_baseline", "persist_eeg_baseline_metrics_closure_v1", "persist_eeg_openbmi_task_generality_v1", "persist_eeg_seven_backbone_fourtask_3seed_v1", "persist_eeg_crossbackbone_peeh_v1", "persist_eeg_protected_complement_coupling_seed0_v1")]
    value = {"schema": "PC_REFINE_EEGNET_V1_SEED0", "tasks": TASKS, "folds": list(FOLDS), "seed": 0,
             "variants": VARIANTS, "insertion": "spatial_elu_pool1_after_drop1", "context_dim": 16, "interaction_dim": 8,
             "contingent_random_followup": "run only for tasks with positive primary final-heldout BA point difference; exploratory post-heldout; use prelocked random basis and Protected-PC selected epoch budgets",
             "phase1": {"epochs": 20, "lr": 3e-4}, "phase2": {"epochs": 10, "adapter_lr": 1e-4, "suffix_lr": 1e-5},
             "weight_decay": 5e-4, "trust_lambda": 1e-4, "pathway_ridge_alpha": 1.0,
             "BN_running_statistics": "frozen",
             "discovery_selection": "subject-equal BA then NLL then earlier epoch", "final_heldout_accessed": False,
             "source_refs": [{"path": str(x.relative_to(REPO)), "exists": x.is_dir()} for x in refs],
             "source_code_sha256": sha(Path(__file__)), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json_write(path, value)
    (PROTOCOL / "PROTOCOL_LOCK.sha256").write_text(sha(path) + "\n", encoding="utf-8")


def preflight() -> None:
    audit = []
    for task in TASKS:
        for fold in FOLDS:
            data = development(task, fold)
            model, rec, ck = load_anchor(task, fold, data)
            audit.append({"task": task, "fold": fold, "selected_epoch": rec["selected_epoch"], "checkpoint_sha256": sha(ck),
                          "split_sha256": data["split"], "normalizer_sha256": data["normalizer"]["mean_std_sha256"],
                          "train_subject_ids_sha256": hashlib.sha256(json.dumps(sorted(set(data["ss"]))).encode()).hexdigest(),
                          "discovery_subject_ids_sha256": hashlib.sha256(json.dumps(sorted(set(data["vs"]))).encode()).hexdigest(),
                          "train_subjects": len(set(data["ss"])), "discovery_subjects": len(set(data["vs"])),
                          "channels": data["channels"], "samples": data["samples"], "classes": data["classes"],
                          "trainable_parameters": sum(p.numel() for p in model.parameters())})
            print("ANCHOR_VERIFIED", task, fold, flush=True)
    csv_write(OUTPUT / "ANCHOR_CHECKPOINT_AUDIT.csv", audit)
    lines = ["# Data protocol audit", "", "Frozen SEARCH splits and task-specific cache loaders are reused.", "",
             "Split file: `experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/protocol/FIVEFOLD_SPLIT.json`.",
             "Formal evaluator reference: `experiments/persist_eeg_baseline_metrics_closure_v1/code/run_frozen_sessions.py`.",
             f"OpenBMI cache: `{OPEN_CACHE}`", f"WBCIC development cache: `{WBCIC_CACHE}`",
             f"WBCIC true-outer cache (metadata only before lock): `{TRUE_WBCIC_CACHE}`", "",
             "OpenBMI physical S1/S2 maps to paper S1/S2. WBCIC physical S0/S1/S2 maps to paper S1/S2/S3.",
             "Preprocessing and label mapping use the frozen `tech_recipe_selection.py` cache loaders; per-channel mean/std uses the `benchmark_data.py` formula on source-session training subjects. OpenBMI ERP retains deterministic class-weighted CE. MI, SSVEP and WBCIC use CE.",
             f"Cache loader SHA256: `{sha(SEVEN_CODE / 'tech_recipe_selection.py')}`; canonical EEGNet SHA256: `{sha(SEVEN_CODE / 'backbone_models.py')}`.",
             "The final evaluator follows baseline metrics closure physical session and subject scope; heldout arrays are loaded only after FINAL_EVAL_LOCK.",
             "Each fold's mean/std is computed on source-session training subjects. The table records the exact hashes.", "",
             "| Task | Fold | Split SHA256 | Train subjects SHA256 | Discovery subjects SHA256 | Normalizer SHA256 | Channels | Samples |", "| --- | ---: | --- | --- | --- | --- | ---: | ---: |"]
    lines += [f"| {r['task']} | {r['fold']} | `{r['split_sha256']}` | `{r['train_subject_ids_sha256']}` | `{r['discovery_subject_ids_sha256']}` | `{r['normalizer_sha256']}` | {r['channels']} | {r['samples']} |" for r in audit]
    manifest = json.loads((REPO / "experiments" / "persist_eeg_final_heldout_confirmation_v1" / "protocol" / "FINAL_HOLDOUT_MANIFEST.json").read_text(encoding="utf-8"))
    open_ids = sorted(map(str, manifest["OpenBMI"]["subject_ids"]), key=int)
    if len(open_ids) != 14 or len(TRUE_WBCIC) != 10: raise RuntimeError("formal heldout scope mismatch")
    lines += ["", f"OpenBMI final 14 subject IDs SHA256: `{hashlib.sha256(json.dumps(open_ids).encode()).hexdigest()}`.",
              f"WBCIC true-outer 10 subject IDs SHA256: `{hashlib.sha256(json.dumps(sorted(TRUE_WBCIC)).encode()).hexdigest()}`."]
    (EXP / "DATA_PROTOCOL_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    protocol_lock()


def discover(task: str, fold: int) -> None:
    target = cell_dir(task, fold, "discovery")
    if (target / "COMPLETE.json").exists(): return
    data = development(task, fold)
    base, rec, ck = load_anchor(task, fold, data)
    basis, record = bases(base, data, task, fold)
    json_write(target / "protected.json", record)
    if record["status"] != "COMPLETE":
        json_write(target / "COMPLETE.json", {"status": "UNDEFINED_PROTECTED", "task": task, "fold": fold}); return
    np.savez_compressed(target / "bases.npz", **basis)
    identity, results, audits = [], [], []
    for variant in VARIANTS[1:]:
        seed_all(stable_seed("adapter-initial", task, fold, variant))
        model = make_variant(base, basis, variant)
        idrow = identity_audit(base, model, data["source"])
        identity.append({"task": task, "fold": fold, "variant": variant, **idrow})
        initial_bn = {k: v.detach().cpu().clone() for k, v in model.base.state_dict().items() if "running_" in k or "num_batches_tracked" in k}
        model, e1, h1 = train_adapter(model, data, task, fold, 1, 20, True, variant)
        model, e2, h2 = train_adapter(model, data, task, fold, 2, 10, True, variant)
        for k, v in initial_bn.items():
            if not torch.equal(v, model.base.state_dict()[k].detach().cpu()): raise RuntimeError("BN running statistics changed")
        ckpath = target / f"{variant}.pt"
        torch_write(ckpath, {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epochs": [e1, e2]})
        met = score(logits(model, data["val"]), data["vy"], data["vs"])
        results.append({"task": task, "fold": fold, "variant": variant, "phase1_selected_epoch": e1,
                        "phase2_selected_epoch": e2, "checkpoint_sha256": sha(ckpath), **met})
        audits.extend([{"task": task, "fold": fold, "variant": variant, **h} for h in h1+h2])
    base_met = score(logits(base, data["val"]), data["vy"], data["vs"])
    results.insert(0, {"task": task, "fold": fold, "variant": "BASELINE", "phase1_selected_epoch": 0,
                       "phase2_selected_epoch": 0, "checkpoint_sha256": sha(ck), **base_met})
    csv_write(target / "ZERO_INIT_IDENTITY_AUDIT.csv", identity)
    csv_write(target / "DISCOVERY_RESULTS.csv", results)
    csv_write(target / "TRAINING_AUDIT.csv", audits)
    json_write(target / "COMPLETE.json", {"status": "COMPLETE", "task": task, "fold": fold,
                                           "protected_rank": record["protected_rank"], "anchor_sha256": sha(ck),
                                           "discovery_results_sha256": sha(target / "DISCOVERY_RESULTS.csv")})


def train_refit_anchor(task: str, fold: int, data: dict[str, Any], epochs: int) -> nn.Module:
    seed_all(0)
    model = eegnet(data)
    seed_all(100000)
    x = torch.from_numpy(np.ascontiguousarray(data["source"])).to(DEVICE)
    y = torch.as_tensor(data["sy"], dtype=torch.long, device=DEVICE)
    weight = None
    if task == "OpenBMI_ERP":
        counts = np.bincount(data["sy"], minlength=data["classes"])
        weight = torch.as_tensor(len(data["sy"]) / (data["classes"] * counts), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    for epoch in range(1, epochs+1):
        model.train()
        # Exact canonical batch-order policy, with enlarged legal refit pool.
        digest = hashlib.sha256(f"SEVEN-BACKBONE|{task}|EEGNet|{fold}|0|{epoch}".encode()).digest()
        permutation = np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(len(y))
        for start in range(0, len(y), 128):
            indices = permutation[start:start+128]
            idx = torch.as_tensor(indices, dtype=torch.long, device=DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.index_select(0, idx)), y.index_select(0, idx), weight=weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        print("REFIT_ANCHOR_EPOCH", task, fold, epoch, flush=True)
    model.eval(); return model


def refit(task: str, fold: int) -> None:
    target = cell_dir(task, fold, "refit")
    if (target / "COMPLETE.json").exists(): return
    discovery_dir = cell_dir(task, fold, "discovery")
    discovery = json.loads((discovery_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    if discovery["status"] != "COMPLETE":
        json_write(target / "COMPLETE.json", {"status": "UNDEFINED_PROTECTED", "task": task, "fold": fold}); return
    data = development(task, fold, refit=True)
    rec, _ = source_anchor(task, fold)
    model = train_refit_anchor(task, fold, data, int(rec["selected_epoch"]))
    basepath = target / "BASELINE.pt"
    torch_write(basepath, {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                           "selected_epoch_budget": int(rec["selected_epoch"])})
    basis, record = bases(model, data, task, fold)
    json_write(target / "protected.json", record)
    if record["status"] != "COMPLETE":
        json_write(target / "COMPLETE.json", {"status": "UNDEFINED_PROTECTED_REFIT", "task": task, "fold": fold,
                                               "baseline_sha256": sha(basepath)}); return
    np.savez_compressed(target / "bases.npz", **basis)
    chosen = {r["variant"]: r for r in csv.DictReader((discovery_dir / "DISCOVERY_RESULTS.csv").open(newline="", encoding="utf-8"))}
    checkpoints = {"BASELINE": sha(basepath)}
    trainrows = []
    for variant in VARIANTS[1:]:
        seed_all(stable_seed("adapter-initial", task, fold, variant))
        adapted = make_variant(model, basis, variant)
        identity_audit(model, adapted, data["source"])
        initial_bn = {k: v.detach().cpu().clone() for k, v in adapted.base.state_dict().items() if "running_" in k or "num_batches_tracked" in k}
        e1 = int(chosen[variant]["phase1_selected_epoch"]); e2 = int(chosen[variant]["phase2_selected_epoch"])
        adapted, _, h1 = train_adapter(adapted, data, task, fold, 1, e1, False, variant)
        adapted, _, h2 = train_adapter(adapted, data, task, fold, 2, e2, False, variant)
        for k, v in initial_bn.items():
            if not torch.equal(v, adapted.base.state_dict()[k].detach().cpu()): raise RuntimeError("refit BN running statistics changed")
        ckpath = target / f"{variant}.pt"
        torch_write(ckpath, {"state_dict": {k: v.detach().cpu() for k, v in adapted.state_dict().items()}, "epochs": [e1,e2]})
        checkpoints[variant] = sha(ckpath)
        trainrows.extend([{"task": task, "fold": fold, "variant": variant, **h} for h in h1+h2])
    csv_write(target / "TRAINING_AUDIT.csv", trainrows)
    json_write(target / "COMPLETE.json", {"status": "COMPLETE", "task": task, "fold": fold,
                                           "subjects": data["subjects"], "subjects_sha256": hashlib.sha256(json.dumps(data["subjects"]).encode()).hexdigest(),
                                           "split_sha256": data["split"], "normalizer_sha256": data["normalizer"]["mean_std_sha256"],
                                           "selected_epochs": {v: [int(chosen[v]["phase1_selected_epoch"]), int(chosen[v]["phase2_selected_epoch"])] for v in VARIANTS[1:]},
                                           "checkpoints": checkpoints, "basis_hashes": {k: arr_sha(basis[k]) for k in ("qp","qr","mu","up","ur")}})


def compile_preheldout_audits() -> None:
    protected, projector_rows, pca_rows, identities, training, discovery_rows, efficiency = [], [], [], [], [], [], []
    anchor_rows = {(r["task"], int(r["fold"])): r for r in csv.DictReader((OUTPUT / "ANCHOR_CHECKPOINT_AUDIT.csv").open(newline="", encoding="utf-8"))}
    for task in TASKS:
        for fold in FOLDS:
            for stage in ("discovery", "refit"):
                directory = cell_dir(task, fold, stage)
                info = json.loads((directory / "protected.json").read_text(encoding="utf-8"))
                if info["status"] != "COMPLETE" or float(info["reconstruction_error"]) >= 1e-5 or int(info["PCA_dim"]) > 16:
                    raise RuntimeError(f"preheldout basis audit failed: {task}/{fold}/{stage}")
                stored = np.load(directory / "bases.npz", allow_pickle=False)
                if info["projector_hash"] != arr_sha(stored["qp"], stored["mu"]) or info["PCA_hash"] != arr_sha(stored["up"]):
                    raise RuntimeError(f"stored basis hash mismatch: {task}/{fold}/{stage}")
                base = {"task": task, "fold": fold, "stage": stage, "status": info["status"]}
                protected.append({**base, "protected_rank": info.get("protected_rank"), "rank": info.get("rank"),
                                  "canonical_basis_hash": info.get("canonical_basis_hash"),
                                  "protected_coordinate_hash": info.get("protected_coordinate_hash"),
                                  "discovery_data_hash": info.get("discovery_data_hash"),
                                  "protected_coordinates": json.dumps(info.get("protected_coordinates", []))})
                projector_rows.append({**base, "projector_rank": info.get("projector_rank"),
                                       "projector_hash": info.get("projector_hash"), "reconstruction_error": info.get("reconstruction_error")})
                pca_rows.append({**base, "PCA_dim": info.get("PCA_dim"), "PCA_hash": info.get("PCA_hash")})
                training += [{"stage": stage, **r} for r in csv.DictReader((directory / "TRAINING_AUDIT.csv").open(newline="", encoding="utf-8"))]
            dd = cell_dir(task, fold, "discovery")
            cell_identities = list(csv.DictReader((dd / "ZERO_INIT_IDENTITY_AUDIT.csv").open(newline="", encoding="utf-8")))
            if len(cell_identities) != 1 or any(float(r["logits_max_abs_diff"]) >= 1e-6 or r["predictions_exact_match"] != "True" or r["BN_state_exact_match"] != "True" for r in cell_identities):
                raise RuntimeError(f"zero-init identity audit failed: {task}/{fold}")
            identities += cell_identities
            discovery_rows += list(csv.DictReader((dd / "DISCOVERY_RESULTS.csv").open(newline="", encoding="utf-8")))
            meta = json.loads((cell_dir(task, fold, "refit") / "protected.json").read_text(encoding="utf-8"))
            a = anchor_rows[task, fold]; samples = int(a["samples"]); channels = int(a["channels"]); classes = int(a["classes"])
            t1, t2 = samples // 4, samples // 4 // 8
            d, r, c = 16*t1, int(meta["projector_rank"]), int(meta["PCA_dim"])
            baseline_macs = 8*channels*samples*64 + 16*channels*samples + 16*t1*16 + 16*16*t1 + 16*t2*64 + 64*classes
            adapter_macs = 3*d*r + d*c + 8*r + 8*c + 8*r
            adapter_params = (r*8+8) + (c*8+8) + (8*r+r)
            base_params = int(a["trainable_parameters"])
            for variant in VARIANTS:
                efficiency.append({"task": task, "fold": fold, "variant": variant,
                                   "adapter_trainable_params": 0 if variant == "BASELINE" else adapter_params,
                                   "total_params": base_params + (0 if variant == "BASELINE" else adapter_params),
                                   "parameter_delta_vs_EEGNet": 0 if variant == "BASELINE" else adapter_params,
                                   "MACs_per_trial": baseline_macs + (0 if variant == "BASELINE" else adapter_macs),
                                   "MAC_convention": "Conv and linear multiply-accumulate pairs; norm/activation excluded"})
    for name, rows in (("PROTECTED_DISCOVERY_AUDIT.csv", protected), ("LAYER_PROJECTOR_AUDIT.csv", projector_rows),
                       ("PCA_CONTEXT_AUDIT.csv", pca_rows), ("ZERO_INIT_IDENTITY_AUDIT.csv", identities),
                       ("TRAINING_AUDIT.csv", training), ("DISCOVERY_RESULTS.csv", discovery_rows), ("EFFICIENCY.csv", efficiency)):
        csv_write(OUTPUT / name, rows)
    json_write(OUTPUT / "ZERO_INIT_IDENTITY_AUDIT.json", {"status": "PASS", "cells": identities})


def final_lock() -> None:
    target = PROTOCOL / "FINAL_EVAL_LOCK.json"
    if target.exists(): raise RuntimeError("final lock already exists")
    rows = []
    for task in TASKS:
        for fold in FOLDS:
            directory = cell_dir(task, fold, "refit")
            cell = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
            if cell["status"] != "COMPLETE": raise RuntimeError(f"refit incomplete: {task}/{fold}: {cell['status']}")
            for variant, expected in cell["checkpoints"].items():
                if sha(directory / f"{variant}.pt") != expected: raise RuntimeError("refit checkpoint hash mismatch")
            stored = np.load(directory / "bases.npz", allow_pickle=False)
            for key, expected in cell["basis_hashes"].items():
                if arr_sha(stored[key]) != expected: raise RuntimeError(f"refit basis hash mismatch: {task}/{fold}/{key}")
            stored.close()
            rows.append({**cell, "bases_npz_sha256": sha(directory / "bases.npz")})
    compile_preheldout_audits()
    code = Path(__file__)
    manifest = json.loads((REPO / "experiments" / "persist_eeg_final_heldout_confirmation_v1" / "protocol" / "FINAL_HOLDOUT_MANIFEST.json").read_text(encoding="utf-8"))
    open_ids = sorted(map(str, manifest["OpenBMI"]["subject_ids"]), key=int)
    if len(open_ids) != 14 or len(TRUE_WBCIC) != 10: raise RuntimeError("final cohort scope mismatch")
    value = {"schema": "PC_REFINE_EEGNET_V1_FINAL_EVAL_LOCK", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "git_commit": os.environ.get("PC_REFINE_GIT_COMMIT", "UNCOMMITTED_WORKTREE"), "code_sha256": sha(code),
             "protocol_amendment_sha256": sha(PROTOCOL / "PROTOCOL_AMENDMENT.md"),
             "contingent_random_followup_code_sha256": sha(code.with_name("exploratory_random.py")),
             "contingent_random_trigger": "per-task positive Protected-PC minus Baseline primary heldout BA; Random-PC remains exploratory post-heldout",
             "model_definitions": {"baseline": "canonical EEGNet from backbone_models.py, refit at selected seed0 epoch",
                                   "protected": "single post-drop1 spatial_elu_pool1 PCAdapter; dC=16, dI=8, zero-initialized W_o; suffix Phase2"},
             "hyperparameters": {"phase1_epochs_max":20,"phase2_epochs_max":10,"phase1_adapter_lr":3e-4,
                                  "phase2_adapter_lr":1e-4,"phase2_suffix_lr":1e-5,"weight_decay":5e-4,
                                  "trust_lambda":1e-4,"batch_size":128,"pathway_ridge_alpha":1.0,
                                  "BN_running_stats":"frozen"},
             "protocol_lock_sha256": sha(PROTOCOL / "PROTOCOL_LOCK.json"), "cells": rows,
             "evaluator": "same canonical task cache loader; five-fold probability mean per biological subject/session/trial; subject-equal BA, Macro-F1, NLL, worst-session BA",
             "evaluator_source_hashes": {str(p.relative_to(REPO)): sha(p) for p in (SEVEN_CODE / "tech_recipe_selection.py", SEVEN_CODE / "backbone_models.py", REPO / "experiments" / "persist_eeg_baseline_metrics_closure_v1" / "code" / "run_frozen_sessions.py")},
             "heldout_subjects": {"OpenBMI": open_ids, "WBCIC_true_outer": list(TRUE_WBCIC)},
             "heldout_subject_id_hashes": {"OpenBMI": hashlib.sha256(json.dumps(open_ids).encode()).hexdigest(),
                                           "WBCIC_true_outer": hashlib.sha256(json.dumps(sorted(TRUE_WBCIC)).encode()).hexdigest()}}
    json_write(target, value)
    (PROTOCOL / "FINAL_EVAL_LOCK.sha256").write_text(sha(target) + "\n", encoding="utf-8")


def check_final_lock() -> dict[str, Any]:
    path = PROTOCOL / "FINAL_EVAL_LOCK.json"
    if not path.is_file() or sha(path) != (PROTOCOL / "FINAL_EVAL_LOCK.sha256").read_text().strip():
        raise RuntimeError("FINAL_EVAL_LOCK missing or modified")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["code_sha256"] != sha(Path(__file__)) or lock["protocol_lock_sha256"] != sha(PROTOCOL / "PROTOCOL_LOCK.json"):
        raise RuntimeError("source changed after final lock")
    if lock["protocol_amendment_sha256"] != sha(PROTOCOL / "PROTOCOL_AMENDMENT.md"):
        raise RuntimeError("protocol amendment changed after final lock")
    if lock["contingent_random_followup_code_sha256"] != sha(Path(__file__).with_name("exploratory_random.py")):
        raise RuntimeError("contingent exploratory plan changed after final lock")
    for p, expected in lock["evaluator_source_hashes"].items():
        if sha(REPO / p) != expected: raise RuntimeError("evaluator changed after final lock")
    for cell in lock["cells"]:
        directory = cell_dir(cell["task"], cell["fold"], "refit")
        if sha(directory / "bases.npz") != cell["bases_npz_sha256"]:
            raise RuntimeError("refit basis file changed after final lock")
        for variant, expected in cell["checkpoints"].items():
            if sha(directory / f"{variant}.pt") != expected:
                raise RuntimeError("refit checkpoint changed after final lock")
    return lock


def final_eval(task: str, fold: int) -> None:
    lock = check_final_lock()  # Must run before final arrays are loaded.
    cell = next(x for x in lock["cells"] if x["task"] == task and x["fold"] == fold)
    target = cell_dir(task, fold, "heldout")
    if (target / "COMPLETE.json").exists(): return
    data = development(task, fold, refit=True)
    if data["normalizer"]["mean_std_sha256"] != cell["normalizer_sha256"]: raise RuntimeError("refit normalizer mismatch")
    ids = list(TRUE_WBCIC) if task == "WBCIC_MI" else list(map(str, json.loads((REPO / "experiments" / "persist_eeg_final_heldout_confirmation_v1" / "protocol" / "FINAL_HOLDOUT_MANIFEST.json").read_text())["OpenBMI"]["subject_ids"]))
    if len(ids) != (10 if task == "WBCIC_MI" else 14): raise RuntimeError("heldout cohort size mismatch")
    expected_ids = lock["heldout_subjects"]["WBCIC_true_outer" if task == "WBCIC_MI" else "OpenBMI"]
    if set(ids) != set(expected_ids): raise RuntimeError("heldout subject IDs differ from final lock")
    if set(ids) & set(data["subjects"]): raise RuntimeError("heldout/development overlap")
    raw = {}
    sessions = (0,1,2) if task == "WBCIC_MI" else (1,2)
    for session in sessions:
        x, y, subjects, _ = rows(task, ids, (session,), data["cache_name"], data["mapping"], final=(task == "WBCIC_MI"))
        raw[session] = (((x - data["mu"][None,:,None]) / np.maximum(data["sd"][None,:,None], 1e-6)).astype(np.float32), y.astype(np.int64), subjects.astype(str))
    directory = cell_dir(task, fold, "refit")
    basis_npz = np.load(directory / "bases.npz", allow_pickle=False)
    basis = {k: basis_npz[k] for k in basis_npz.files}
    output = {}
    for variant in VARIANTS:
        base = eegnet(data)
        if variant == "BASELINE":
            model = base
        else:
            model = make_variant(base, basis, variant)
        ck = directory / f"{variant}.pt"
        if sha(ck) != cell["checkpoints"][variant]: raise RuntimeError("checkpoint changed after lock")
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"], strict=True)
        model.eval()
        for session, (x,y,s) in raw.items():
            z, mechanism = infer_mechanism(model, x, basis, variant)
            output[f"{variant}_S{session}_z"] = z
            for key, value in mechanism.items():
                output[f"{variant}_S{session}_{key}"] = value
            output[f"S{session}_y"] = y; output[f"S{session}_subjects"] = s.astype("U")
        print("FINAL_MODEL_DONE", task, fold, variant, flush=True)
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "predictions.npz", **output)
    json_write(target / "COMPLETE.json", {"task": task, "fold": fold, "status": "COMPLETE", "subjects": ids,
                                           "predictions_sha256": sha(target / "predictions.npz"), "final_lock_sha256": sha(PROTOCOL / "FINAL_EVAL_LOCK.json")})


def paired_ci(a: np.ndarray, b: np.ndarray, tag: tuple[Any, ...]) -> tuple[float,float,float]:
    d = a - b
    rng = np.random.default_rng(stable_seed("paired-bootstrap", *tag))
    idx = rng.integers(0, len(d), size=(20000, len(d)))
    boot = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def aggregate() -> None:
    check_final_lock()
    subject_rows, summary_rows, contrasts, rescue, mechanism_rows = [], [], [], [], []
    for task in TASKS:
        parts = []
        for fold in FOLDS:
            target = cell_dir(task, fold, "heldout")
            rec = json.loads((target / "COMPLETE.json").read_text(encoding="utf-8"))
            if rec["status"] != "COMPLETE" or sha(target / "predictions.npz") != rec["predictions_sha256"]: raise RuntimeError("heldout cell incomplete")
            parts.append(np.load(target / "predictions.npz", allow_pickle=False))
        sessions = (0,1,2) if task == "WBCIC_MI" else (1,2)
        session_subject = {}
        for session in sessions:
            y = parts[0][f"S{session}_y"]
            subjects = parts[0][f"S{session}_subjects"].astype(str)
            for p in parts[1:]:
                if not np.array_equal(y, p[f"S{session}_y"]) or not np.array_equal(subjects, p[f"S{session}_subjects"].astype(str)):
                    raise RuntimeError("fold trial order mismatch")
            probabilities = {}
            for variant in VARIANTS:
                z = np.stack([p[f"{variant}_S{session}_z"] for p in parts])
                prob = np.exp(z - z.max(axis=2, keepdims=True)); prob /= prob.sum(axis=2, keepdims=True)
                probabilities[variant] = prob.mean(0)
            for sub in sorted(set(subjects), key=lambda t: int(t.replace("sub-", ""))):
                ix = subjects == sub
                session_subject[sub, session] = {}
                for variant in VARIANTS:
                    prob = probabilities[variant][ix]
                    pred = prob.argmax(1)
                    row = {"task": task, "subject": sub, "session": f"S{session}", "variant": variant,
                           "BA": float(balanced_accuracy_score(y[ix], pred)),
                           "macro_F1": float(f1_score(y[ix], pred, average="macro", zero_division=0)),
                           "NLL": float(-np.log(np.clip(prob[np.arange(len(prob)), y[ix]], 1e-12, 1)).mean())}
                    subject_rows.append(row); session_subject[sub, session][variant] = row
                bp = probabilities["BASELINE"][ix].argmax(1)
                pp = probabilities["PROTECTED_PC_REFINE"][ix].argmax(1)
                mechanism = {"task": task, "subject": sub, "session": f"S{session}"}
                for variant in VARIANTS:
                    for key in ("P_logit_rms", "C_logit_rms", "PC_disagreement"):
                        mechanism[f"{variant}_{key}"] = float(np.mean([p[f"{variant}_S{session}_{key}"][ix].mean() for p in parts]))
                for key in ("correction_norm", "relative_correction"):
                    mechanism[key] = float(np.mean([p[f"PROTECTED_PC_REFINE_S{session}_{key}"][ix].mean() for p in parts]))
                energy = np.mean([p[f"PROTECTED_PC_REFINE_S{session}_delta_sq"][ix].mean(0) for p in parts], axis=0)
                mechanism["top3_correction_energy_share"] = float(np.sort(energy)[-min(3,len(energy)):].sum() / max(float(energy.sum()), 1e-12))
                mechanism_rows.append(mechanism)
                rescue.append({**mechanism, "baseline_errors_rescued": int(((bp != y[ix]) & (pp == y[ix])).sum()),
                               "baseline_correct_damaged": int(((bp == y[ix]) & (pp != y[ix])).sum()), "trials": int(ix.sum())})
        primary = 2
        subjects = sorted({s for s,_ in session_subject}, key=lambda t: int(t.replace("sub-", "")))
        if len(subjects) != (10 if task == "WBCIC_MI" else 14): raise RuntimeError("heldout subject count mismatch")
        vec = {}
        for variant in VARIANTS:
            for metric in ("BA", "macro_F1"):
                vec[variant, metric] = np.asarray([session_subject[s, primary][variant][metric] for s in subjects])
            worst = np.asarray([min(session_subject[s, ses][variant]["BA"] for ses in sessions) for s in subjects])
            vec[variant, "worst_session_BA"] = worst
            summary_rows.append({"task": task, "variant": variant, "subjects": len(subjects), "primary_session": f"S{primary}",
                                 "BA": float(vec[variant, "BA"].mean()), "macro_F1": float(vec[variant, "macro_F1"].mean()),
                                 "worst_session_BA": float(worst.mean()),
                                 "NLL": float(np.mean([session_subject[s, primary][variant]["NLL"] for s in subjects]))})
        for comparator in VARIANTS[:-1]:
            for metric in ("BA", "macro_F1", "worst_session_BA"):
                d, lo, hi = paired_ci(vec["PROTECTED_PC_REFINE", metric], vec[comparator, metric], (task, comparator, metric))
                contrasts.append({"task": task, "contrast": f"PROTECTED_PC_REFINE - {comparator}", "metric": metric,
                                  "difference": d, "CI95_low": lo, "CI95_high": hi, "bootstrap_draws": 20000, "unit": "biological subject"})
        for p in parts: p.close()
    csv_write(OUTPUT / "HELDOUT_SUBJECT_RESULTS.csv", subject_rows)
    session_rows = []
    for task in TASKS:
        for session in ((0,1,2) if task == "WBCIC_MI" else (1,2)):
            for variant in VARIANTS:
                group = [r for r in subject_rows if r["task"] == task and r["session"] == f"S{session}" and r["variant"] == variant]
                expected = 10 if task == "WBCIC_MI" else 14
                if len(group) != expected: raise RuntimeError("session summary subject count mismatch")
                session_rows.append({"task": task, "session": f"S{session}", "variant": variant,
                                     "subjects": expected, **{metric: float(np.mean([r[metric] for r in group]))
                                                            for metric in ("BA", "macro_F1", "NLL")}})
    csv_write(OUTPUT / "HELDOUT_SESSION_SUMMARY.csv", session_rows)
    csv_write(OUTPUT / "HELDOUT_MODEL_TASK_SUMMARY.csv", summary_rows)
    csv_write(OUTPUT / "PAIRED_HELDOUT_CONTRASTS.csv", contrasts)
    csv_write(OUTPUT / "HELDOUT_RESCUE_HARM.csv", rescue)
    csv_write(OUTPUT / "HELDOUT_MECHANISM_AUDIT.csv", mechanism_rows)
    summary = {(r["task"],r["variant"]):r for r in summary_rows}
    lines = ["# PC-Refine EEGNet V1 final report", "", "Seed 0; five-fold probability mean; formal final-heldout subjects; primary future session.", "",
             "| Task | Baseline BA | Protected-PC BA | Δ BA vs Baseline |",
             "| --- | ---: | ---: | ---: |"]
    for task in TASKS:
        b,v = [summary[task, name]["BA"] for name in VARIANTS]
        lines.append(f"| {task} | {b:.4f} | {v:.4f} | {v-b:+.4f} |")
    lines += ["", "## Paired heldout differences", "",
              "Biological-subject bootstrap, 20,000 draws. Units are absolute metric points.", "",
              "| Task | Metric | Difference | 95% CI |", "| --- | --- | ---: | ---: |"]
    for row in contrasts:
        lines.append(f"| {row['task']} | {row['metric']} | {row['difference']:+.4f} | [{row['CI95_low']:+.4f}, {row['CI95_high']:+.4f}] |")
    lines += ["", "## Physical-session subject-equal BA", "",
              "OpenBMI physical S1/S2; WBCIC physical S0/S1/S2 corresponds to paper S1/S2/S3.", "",
              "| Task | Session | Baseline BA | Protected-PC BA |", "| --- | --- | ---: | ---: |"]
    by_session = {(r["task"], r["session"], r["variant"]): r for r in session_rows}
    for task in TASKS:
        for session in ((0,1,2) if task == "WBCIC_MI" else (1,2)):
            label = f"S{session}"
            b = by_session[task, label, "BASELINE"]["BA"]
            p = by_session[task, label, "PROTECTED_PC_REFINE"]["BA"]
            lines.append(f"| {task} | {label} | {b:.4f} | {p:.4f} |")
    gains = [summary[task, "PROTECTED_PC_REFINE"]["BA"] - summary[task, "BASELINE"]["BA"] for task in TASKS]
    lines += ["", "## Interpretation", "", f"Protected-PC has a positive heldout BA difference on {sum(x > 0 for x in gains)}/4 tasks.",
              "The user narrowed this run to the modified model and its matched seed-0 baseline. This comparison cannot isolate a mechanism-specific gain from a generic adapter effect.",
              "Random-PC is a separate exploratory follow-up only for tasks with a positive primary heldout BA difference."]
    (OUTPUT / "FINAL_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "discover", "refit", "lock", "final-eval", "aggregate"))
    parser.add_argument("--task", choices=TASKS); parser.add_argument("--fold", type=int, choices=FOLDS)
    a = parser.parse_args()
    if a.stage in ("discover", "refit", "final-eval") and (a.task is None or a.fold is None): parser.error("task and fold required")
    {"preflight": preflight, "discover": lambda: discover(a.task,a.fold), "refit": lambda: refit(a.task,a.fold),
     "lock": final_lock, "final-eval": lambda: final_eval(a.task,a.fold), "aggregate": aggregate}[a.stage]()


if __name__ == "__main__": main()
