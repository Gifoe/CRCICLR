"""Frozen seed-0 cross-backbone PEEH audit (TFFormer and incomplete SGN excluded).

No model parameters or BN buffers are updated.  Runtime cell JSON files are
resume-safe and live outside git; deliverables are compact CSV/JSON/Markdown.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score


EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "outputs" / "crossbackbone_peeh_v1"
PROTOCOL = EXP / "protocol"
RUNTIME = Path(os.environ.get("PEEH_RUNTIME", r"D:\nips-temp\TotalP\P1\crossbackbone_peeh_runtime"))
SEVEN_REPO = Path(os.environ.get("SEVEN_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK"))
SEVEN_CODE = SEVEN_REPO / "experiments" / "persist_eeg_seven_backbone_fourtask_3seed_v1" / "code"
SEVEN_RUNTIME = Path(os.environ.get("SEVEN_RUNTIME", r"D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime"))
OPENBMI_CACHE = Path(os.environ.get("FULL_OPENBMI_CACHE", r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi"))
WBCIC_CACHE = Path(os.environ.get("FULL_WBCIC_CACHE", r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache\wbcic_epochs"))
TRUE_WBCIC_CACHE = Path(os.environ.get("TRUE_OUTER_WBCIC_CACHE", r"D:\nips-temp\TotalP\P2\wbcic_outer_cache\wbcic_epochs"))
HOLDOUT_MANIFEST = SEVEN_REPO / "experiments" / "persist_eeg_final_heldout_confirmation_v1" / "protocol" / "FINAL_HOLDOUT_MANIFEST.json"

RECENT = {
    "ModernTCN": (Path(r"D:\nips-temp\TotalP\P1\CRCICLR_MODERNTCN_FINAL\experiments\persist_eeg_moderntcn_4task_3seed_final_v1\code"), Path(r"D:\nips-temp\TotalP\P1\baseline3_runtime\moderntcn")),
    "Medformer": (Path(r"D:\nips-temp\TotalP\P1\CRCICLR_MEDFORMER_FINAL\experiments\persist_eeg_medformer_4task_3seed_final_v1\code"), Path(r"D:\nips-temp\TotalP\P1\baseline3_runtime\medformer")),
}
MODELS = ("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
FOLDS = tuple(range(5))
SEED = 0
CAP = 32
ACTIVE_RANK = 20
PERSISTENCE_PERMUTATIONS = 200
UTILITY_SPLITS = 5
RANDOM_ERASURES = 100
BOOTSTRAP_DRAWS = 20_000
RIDGE_ALPHA = 0.01
TRUE_WBCIC_SUBJECTS = ("sub-4", "sub-8", "sub-10", "sub-15", "sub-20", "sub-39", "sub-40", "sub-43", "sub-46", "sub-51")

if str(SEVEN_CODE) not in sys.path:
    sys.path.insert(0, str(SEVEN_CODE))
os.environ.setdefault("SEVEN_REPO", str(SEVEN_REPO))
os.environ.setdefault("OFFICIAL_BACKBONE_ROOT", str(SEVEN_RUNTIME / "official"))
os.environ.setdefault("FULL_OPENBMI_CACHE", str(OPENBMI_CACHE))
os.environ.setdefault("FULL_WBCIC_CACHE", str(WBCIC_CACHE))
os.environ.setdefault("MODERN_REPO", str(SEVEN_REPO))
os.environ.setdefault("TASK_GENERALITY_REPO", str(SEVEN_REPO))

_MODULES: dict[str, Any] = {}


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32 - 1)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(v: Any) -> Any:
    if isinstance(v, Path): return str(v)
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows([{k: clean(row.get(k, "")) for k in fields} for row in rows])
    os.replace(tmp, path)


def module(name: str, path: Path) -> Any:
    key = f"{name}:{path}"
    if key in _MODULES: return _MODULES[key]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    _MODULES[key] = value
    return value


def natural_subjects(values: Iterable[object]) -> list[str]:
    def key(x: str) -> tuple[int, str]:
        digits = "".join(c for c in x if c.isdigit())
        return (int(digits) if digits else 10**9, x)
    return sorted({str(x) for x in values}, key=key)


def cell(model: str, task: str, fold: int) -> tuple[dict[str, Any], Path]:
    if model in ("EEGNet", "CBraMod", "TeCh"):
        root = SEVEN_RUNTIME / "search_cells" / task.lower() / model.lower().replace("-", "_") / f"fold{fold}_seed0"
    else:
        root = RECENT[model][1] / "cells" / task.lower() / f"fold{fold}_seed0"
    record_path, ckpt = root / "record.json", root / "selected.pt"
    if not record_path.is_file() or not ckpt.is_file(): raise FileNotFoundError(root)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    identity = (record.get("model"), record.get("task"), int(record.get("fold", -1)), int(record.get("seed", -1)))
    if identity != (model, task, fold, 0): raise RuntimeError(f"identity mismatch {root}: {identity}")
    actual = sha(ckpt)
    if record.get("checkpoint_sha256") != actual: raise RuntimeError(f"checkpoint hash mismatch {ckpt}")
    return record, ckpt


def all_checkpoint_rows() -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for task in TASKS:
            for fold in FOLDS:
                record, ckpt = cell(model, task, fold)
                rows.append({"Model": model, "Task": task, "fold": fold, "seed": 0,
                             "checkpoint_path": str(ckpt), "checkpoint_sha256": sha(ckpt),
                             "selected_epoch": int(record["selected_epoch"]), "split_sha256": record["split_sha256"],
                             "normalizer_sha256": record["normalizer"]["mean_std_sha256"],
                             "trainable_parameters": int(record.get("trainable_parameters", record.get("parameters"))),
                             "recipe_name": record.get("recipe", {}).get("name"),
                             "channels": int(record.get("channels") or (58 if task == "WBCIC_MI" else 62)),
                             "samples": int(record.get("samples") or (250 if task == "OpenBMI_ERP" else 1000)),
                             "classes": int(record["classes"])})
    if len(rows) != 100: raise RuntimeError(f"expected 100 frozen cells, found {len(rows)}")
    return rows


def prelock() -> None:
    rows = all_checkpoint_rows()
    lock = {"schema": "PERSIST_EEG_CROSSBACKBONE_PEEH_LOCK_V1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": list(MODELS), "excluded_models": {"TFFormer": "user excluded", "SGN": "6/20 seed0 checkpoints missing; no training allowed"},
            "tasks": list(TASKS), "folds": list(FOLDS), "seed": 0, "checkpoint_count": len(rows), "checkpoints": rows,
            "protocol": {"rank_threshold": "max(1e-3*lambda_max,1e-8)", "active_rank": 20,
                         "eigen_floor": "max(1e-4*lambda_r,1e-8)", "max_block_size": 4,
                         "persistence_permutations": 200, "utility_subject_half_splits": 5,
                         "protected_rule": "persistence supported AND lower95 absolute CE harm > 0 AND lower95 excess CE harm > 0",
                         "ridge_alpha": RIDGE_ALPHA, "random_erasures": RANDOM_ERASURES,
                         "trial_cap_per_subject_session_class": CAP,
                         "probe_standardizer": "fit on intact TRAIN representation and held fixed across interventions; every intervention refits ridge coefficients",
                         "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_unit": "biological subject"},
            "outcome_isolation": {"evaluation_subjects_used_for_geometry": False, "evaluation_subjects_used_for_ridge_fit": False,
                                  "evaluation_subjects_used_for_random_control": False}}
    write_json(PROTOCOL / "PEEH_PROTOCOL_LOCK.json", lock)
    (PROTOCOL / "PEEH_PROTOCOL_LOCK.sha256").write_text(sha(PROTOCOL / "PEEH_PROTOCOL_LOCK.json") + "\n", encoding="utf-8")
    write_csv(OUT / "CHECKPOINT_AUDIT.csv", rows)
    print("PEEH_PROTOCOL_LOCKED_100_FROZEN_RUNS", flush=True)


def load_fold_arrays(task: str, fold: int) -> dict[str, Any]:
    data_code = module("peeh_benchmark_data", SEVEN_CODE / "benchmark_data.py")
    inner = data_code._load_module("peeh_inner_loader", SEVEN_CODE / "tech_recipe_selection.py")
    modern, task_module = data_code._sources()
    manifest = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    if task in ("OpenBMI_MI", "WBCIC_MI"):
        dataset = "OpenBMI" if task == "OpenBMI_MI" else "WBCIC"
        folds, _, _ = modern.load_split(); role = next(x for x in folds[dataset] if int(x["fold_id"]) == fold)
        train_subjects = list(map(str, role["inner_train_subjects"])); source_sessions = tuple(modern.SOURCE_SESSIONS[dataset]); future_session = int(modern.EVAL_SESSION)
        if dataset == "OpenBMI":
            src, sy, ss, mapping = inner._openbmi_rows(OPENBMI_CACHE, train_subjects, source_sessions, "mi")
            fut, fy, fs, _ = inner._openbmi_rows(OPENBMI_CACHE, train_subjects, (future_session,), "mi", mapping)
            hs = list(map(str, manifest["OpenBMI"]["subject_ids"])); held, hy, hsub, _ = inner._openbmi_rows(OPENBMI_CACHE, hs, (future_session,), "mi", mapping)
        else:
            src, sy, ss, mapping = inner._wbcic_rows(WBCIC_CACHE, train_subjects, source_sessions)
            fut, fy, fs, _ = inner._wbcic_rows(WBCIC_CACHE, train_subjects, (future_session,), mapping)
            held, hy, hsub, _ = inner._wbcic_rows(TRUE_WBCIC_CACHE, list(TRUE_WBCIC_SUBJECTS), (future_session,), mapping)
    else:
        dataset = "OpenBMI"; name = "ERP" if task == "OpenBMI_ERP" else "SSVEP"
        _, _, reference, _ = task_module.split_reference(); role = next(x for x in reference["folds"] if int(x["fold_id"]) == fold)
        spec = task_module.TASKS[name]; train_subjects = list(map(str, role["inner_train_subjects"])); source_sessions = (int(spec["source_session"]),); future_session = int(spec["future_session"])
        src, sy, ss, mapping = inner._openbmi_rows(OPENBMI_CACHE, train_subjects, source_sessions, spec["cache_name"])
        fut, fy, fs, _ = inner._openbmi_rows(OPENBMI_CACHE, train_subjects, (future_session,), spec["cache_name"], mapping)
        hs = list(map(str, manifest["OpenBMI"]["subject_ids"])); held, hy, hsub, _ = inner._openbmi_rows(OPENBMI_CACHE, hs, (future_session,), spec["cache_name"], mapping)
    src, [fut, held], norm = data_code._normalise(src, fut, held)
    classes = int(max(np.max(sy), np.max(fy), np.max(hy)) + 1)
    return {"dataset": dataset, "train_subjects": train_subjects, "source_session": int(source_sessions[0]), "future_session": future_session,
            "src": src, "sy": sy.astype(np.int64), "ss": ss.astype(str), "fut": fut, "fy": fy.astype(np.int64), "fs": fs.astype(str),
            "held": held, "hy": hy.astype(np.int64), "hsub": hsub.astype(str), "classes": classes, "normalizer": norm}


def capped_indices(subjects: np.ndarray, labels: np.ndarray, session: int, task: str, fold: int, purpose: str) -> np.ndarray:
    chosen: list[int] = []
    for subject in natural_subjects(subjects):
        for label in sorted(np.unique(labels).astype(int)):
            idx = np.flatnonzero((subjects.astype(str) == subject) & (labels == label))
            if len(idx) > CAP:
                idx = np.sort(np.random.default_rng(stable_seed("cap", task, fold, purpose, subject, session, label, CAP)).choice(idx, CAP, replace=False))
            chosen.extend(map(int, idx))
    return np.asarray(sorted(chosen), dtype=np.int64)


def prepare_capped(data: dict[str, Any], task: str, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = capped_indices(data["ss"], data["sy"], data["source_session"], task, fold, "train-source")
    b = capped_indices(data["fs"], data["fy"], data["future_session"], task, fold, "train-future")
    e = capped_indices(data["hsub"], data["hy"], data["future_session"], task, fold, "evaluation")
    x = np.concatenate([data["src"][a], data["fut"][b]])
    y = np.concatenate([data["sy"][a], data["fy"][b]])
    s = np.concatenate([data["ss"][a], data["fs"][b]])
    se = np.concatenate([np.full(len(a), data["source_session"]), np.full(len(b), data["future_session"])]).astype(np.int64)
    return x, y, s, se, data["held"][e], data["hy"][e], data["hsub"][e]


def purge_vendor_namespaces() -> None:
    for ns in ("models", "layers", "utils"):
        for key in [x for x in sys.modules if x == ns or x.startswith(ns + ".")]: sys.modules.pop(key, None)


def build_model(row: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    model_name = row["Model"]
    if model_name in ("EEGNet", "CBraMod", "TeCh"):
        models = module("peeh_old_models", SEVEN_CODE / "backbone_models.py")
        model = models.build_model(model_name, dataset="WBCIC" if row["Task"] == "WBCIC_MI" else "OpenBMI",
                                   channels=row["channels"], samples=row["samples"], classes=row["classes"],
                                   tech_recipe=row["recipe_name"] if model_name == "TeCh" else None)
    else:
        purge_vendor_namespaces(); code = RECENT[model_name][0]
        if str(code) not in sys.path: sys.path.insert(0, str(code))
        adapter = module(f"peeh_{model_name.lower()}_{row['Task']}_{row['fold']}", code / "model_adapter.py")
        model = adapter.build_model(channels=row["channels"], samples=row["samples"], classes=row["classes"])
    payload = torch.load(row["checkpoint_path"], map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload), strict=True); model.eval().to(device)
    for p in model.parameters(): p.requires_grad_(False)
    if model_name in ("EEGNet", "CBraMod"): head = model.head
    elif model_name == "TeCh": head = model.model.projector
    elif model_name == "ModernTCN": head = model.model.model.head_class
    elif model_name == "Medformer": head = model.model.projection
    else: raise KeyError(model_name)
    return model, head


def resample(model_name: str, x: np.ndarray) -> np.ndarray:
    if model_name in ("CBraMod",):
        runner = module("peeh_old_runner", SEVEN_CODE / "run_search.py")
        return runner._resample(model_name, x)
    return np.ascontiguousarray(x, dtype=np.float32)


def representations(model: torch.nn.Module, head: torch.nn.Module, x: np.ndarray, model_name: str, device: torch.device, batch: int = 64) -> np.ndarray:
    parts: list[np.ndarray] = []; captured: list[torch.Tensor] = []
    handle = head.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
    try:
        value = resample(model_name, x)
        with torch.inference_mode():
            for start in range(0, len(value), batch):
                captured.clear(); model(torch.from_numpy(value[start:start+batch]).to(device))
                if len(captured) != 1: raise RuntimeError("classifier hook did not fire exactly once")
                parts.append(captured[0].reshape(captured[0].shape[0], -1).float().cpu().numpy())
    finally: handle.remove()
    return np.concatenate(parts).astype(np.float32, copy=False)


def spectrum(h: np.ndarray, y: np.ndarray, subjects: np.ndarray, sessions: np.ndarray, task: str, model: str, fold: int) -> dict[str, Any]:
    # A fixed-seed randomized SVD is used only as a numerically efficient
    # eigensolver for the top active subspace; no projection dimension is
    # imposed on the input representation D.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(h).to(device); mu = x.mean(0); xc = x - mu
    q = min(32, xc.shape[0]-1, xc.shape[1]); torch.manual_seed(stable_seed("svd", model, task, fold))
    _, sv, V = torch.pca_lowrank(xc, q=q, center=False, niter=6)
    ev = (sv.square() / max(len(h)-1, 1)).double(); threshold = max(float(ev[0]) * 1e-3, 1e-8)
    numerical_rank_lower_bound = int((ev > threshold).sum()); r = min(ACTIVE_RANK, numerical_rank_lower_bound)
    if r < 4: raise RuntimeError(f"active rank below 4: {model}/{task}/f{fold}")
    floor = max(float(ev[r-1]) * 1e-4, 1e-8); active = torch.clamp(ev[:r], min=floor).float(); basis = V[:, :r]
    z = (xc @ basis / torch.sqrt(active)).cpu().numpy(); basis_np = basis.cpu().numpy(); mu_np = mu.cpu().numpy()
    del x, xc, V, basis; torch.cuda.empty_cache() if device.type == "cuda" else None
    sess = sorted(np.unique(sessions).astype(int).tolist()); labels = sorted(np.unique(y).astype(int).tolist())
    cent: dict[tuple[str,int,int], np.ndarray] = {}
    for s in natural_subjects(subjects):
        for se in sess:
            for lab in labels:
                idx = np.flatnonzero((subjects.astype(str)==s)&(sessions==se)&(y==lab))
                if len(idx): cent[(s,se,lab)] = z[idx].mean(0)
    subs = natural_subjects(subjects); ops=[]
    for lab in labels:
        left=[]; right=[]
        for s in subs:
            if (s,sess[0],lab) in cent and (s,sess[1],lab) in cent:
                left.append(cent[(s,sess[0],lab)]); right.append(cent[(s,sess[1],lab)])
        if len(left)>=3:
            a=np.asarray(left); b=np.asarray(right); a-=a.mean(0); b-=b.mean(0); ops.append((a.T@b+b.T@a)/(2*len(a)))
    C=np.mean(ops,axis=0); rho,directions=np.linalg.eigh((C+C.T)/2); order=np.argsort(rho)[::-1]; rho=rho[order]; directions=directions[:,order]
    gaps=np.abs(np.diff(rho)); gap_threshold=max(float(np.median(gaps)*4),float(np.max(np.abs(rho))*.05),1e-10)
    cuts=[0]+[i+1 for i,g in enumerate(gaps) if g>gap_threshold]+[r]; blocks=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        for start in range(a,b,4): blocks.append(list(range(start,min(start+4,b))))
    if len(blocks)<2: blocks=[list(range(0,min(4,r))),list(range(min(4,r),r))]
    rng=np.random.default_rng(stable_seed("persistence-null",model,task,fold)); null=[[] for _ in blocks]
    for _ in range(PERSISTENCE_PERMUTATIONS):
        perm=rng.permutation(len(subs)); class_ops=[]
        for lab in labels:
            left=[];right=[]
            for i,s in enumerate(subs):
                t=subs[perm[i]]
                if (s,sess[0],lab) in cent and (t,sess[1],lab) in cent:
                    left.append(cent[(s,sess[0],lab)]);right.append(cent[(t,sess[1],lab)])
            if len(left)>=3:
                a=np.asarray(left);b=np.asarray(right);a-=a.mean(0);b-=b.mean(0);class_ops.append((a.T@b+b.T@a)/(2*len(a)))
        if class_ops:
            cn=np.mean(class_ops,axis=0)
            for bi,block in enumerate(blocks): null[bi].append(float(np.mean(np.diag(directions[:,block].T@cn@directions[:,block]))))
    support=[]
    for bi,block in enumerate(blocks):
        nv=np.asarray(null[bi]); obs=float(np.mean(rho[block])); p95=float(np.quantile(nv,.95))
        support.append({"block":bi,"dimensions":len(block),"rho":obs,"null_p95":p95,"persistence_supported":bool(obs>p95)})
    # Original-D erasure basis: q=(h-mu)@basis/sqrt(ev), then subtract q@basis.T*sqrt(ev).
    return {"mean":mu_np,"basis":basis_np,"scale":np.sqrt(active.cpu().numpy()),"directions":directions.astype(np.float32),
            "rho":rho.astype(np.float32),"blocks":blocks,"support":support,"rank":r,"numerical_rank_lower_bound":numerical_rank_lower_bound,
            "representation_dim":int(h.shape[1]),"svd_solver":"torch.pca_lowrank(q=32,niter=6,fixed-seed)"}


def canonical(h: np.ndarray, spec: dict[str,Any]) -> np.ndarray:
    return ((h-spec["mean"])@spec["basis"]/spec["scale"])@spec["directions"]


def erase(h: np.ndarray, spec: dict[str,Any], dims: Sequence[int]) -> np.ndarray:
    if len(dims) == 0: return h
    q=canonical(h,spec); dirs=spec["directions"][:,np.asarray(dims)]; base=(dirs.T*spec["scale"][None,:])@spec["basis"].T
    return (h-q[:,np.asarray(dims)]@base).astype(np.float32)


def _erasure_base(spec: dict[str, Any]) -> np.ndarray:
    """Return raw-representation erasure rows for every active direction."""
    return ((spec["directions"].T * spec["scale"][None, :]) @ spec["basis"].T).astype(np.float32)


class FixedStandardizerKernelRidge:
    """Exact low-rank intervention path for the locked ridge protocol.

    The mean and standard deviation are fitted once on the intact training
    representation.  If A contains standardized erasure rows and Q contains
    canonical coordinates, an intervention has X' = X - Q A.  Consequently

      X'X'^T = K - UQ^T - QU^T + QGQ^T,

    where U=XA^T and G=AA^T.  The change to K is rank at most 2*len(dims), so
    Woodbury reuses the intact kernel solve.  No intervention constructs an
    N-by-D erased representation or another D-dimensional Gram product.
    """

    def __init__(self, hfit: np.ndarray, yfit: np.ndarray, heval: np.ndarray,
                 classes: int, spec: dict[str, Any], *,
                 qfit: np.ndarray | None = None, qeval: np.ndarray | None = None,
                 raw_base: np.ndarray | None = None) -> None:
        self.classes = int(classes)
        self.rank = int(spec["rank"])
        self.mu = hfit.mean(0, dtype=np.float64)
        self.sd = hfit.std(0, dtype=np.float64)
        self.sd[self.sd < 1e-6] = 1.0

        # Retain float32 feature products, as in the original runner, then use
        # float64 for solves and low-rank updates.
        X = ((hfit - self.mu) / self.sd).astype(np.float32)
        Z = ((heval - self.mu) / self.sd).astype(np.float32)
        base = _erasure_base(spec) if raw_base is None else raw_base
        A = (base / self.sd[None, :]).astype(np.float32)
        Q = canonical(hfit, spec) if qfit is None else qfit
        R = canonical(heval, spec) if qeval is None else qeval
        Q = np.asarray(Q, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)
        Y = np.eye(self.classes, dtype=np.float64)[yfit]
        self.ym = Y.mean(0)
        Yc = Y - self.ym

        # Compact backbones are faster in the direct primal system.  The
        # catastrophic case is D >> N (e.g. ModernTCN D=230,144), where the
        # dual low-rank path below is required.
        self.mode = "primal" if X.shape[1] <= X.shape[0] else "dual_woodbury"
        if self.mode == "primal":
            self.X = X
            self.Z = Z
            self.A = A
            self.Q = Q
            self.R = R
            self.Yc = Yc
            return

        K = (X @ X.T).astype(np.float64)
        self.L = (Z @ X.T).astype(np.float64)
        self.U = (X @ A.T).astype(np.float64)
        self.V = (Z @ A.T).astype(np.float64)
        self.G = (A @ A.T).astype(np.float64)
        self.Q = Q.astype(np.float64)
        self.R = R.astype(np.float64)

        H = K + RIDGE_ALPHA * np.eye(len(K), dtype=np.float64)
        # All active directions are solved together.  Each intervention then
        # selects at most 2*rank columns and solves only its small Woodbury core.
        Fall = np.concatenate([self.U, self.Q], axis=1)
        solved = np.linalg.solve(H, np.concatenate([Yc, Fall], axis=1))
        self.c0 = solved[:, :self.classes]
        self.HF = solved[:, self.classes:]
        self.F = Fall
        self.Fc0 = Fall.T @ self.c0
        self.FHF = Fall.T @ self.HF
        self.Lc0 = self.L @ self.c0
        self.LHF = self.L @ self.HF
        self.K = K
        self.Yc = Yc

    def scores(self, dims: Sequence[int]) -> np.ndarray:
        dims = np.asarray(dims, dtype=np.int64)
        if dims.ndim != 1 or np.any(dims < 0) or np.any(dims >= self.rank):
            raise ValueError(f"invalid erasure dimensions: {dims}")
        if self.mode == "primal":
            if len(dims):
                X = self.X - self.Q[:, dims] @ self.A[dims]
                Z = self.Z - self.R[:, dims] @ self.A[dims]
            else:
                X, Z = self.X, self.Z
            W = np.linalg.solve(
                X.T @ X + RIDGE_ALPHA * np.eye(X.shape[1]), X.T @ self.Yc
            )
            return Z @ W + self.ym
        if len(dims) == 0:
            return self.Lc0 + self.ym
        d = len(dims)
        take = np.concatenate([dims, self.rank + dims])
        G = self.G[np.ix_(dims, dims)]
        # C^-1 for C=[[0,-I],[-I,G]].
        Cinv = np.block([[-G, -np.eye(d)], [-np.eye(d), np.zeros((d, d))]])
        M = Cinv + self.FHF[np.ix_(take, take)]
        rhs = self.Fc0[take]
        use_fallback = not np.all(np.isfinite(M))
        if not use_fallback:
            try:
                # Cancellation in Woodbury can make the small core ill
                # conditioned even though the alpha-regularized updated kernel
                # is positive definite.  The fallback still avoids all D work.
                use_fallback = np.linalg.cond(M) > 1e12
                if not use_fallback:
                    w = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                use_fallback = True
        if use_fallback:
            Q = self.Q[:, dims]
            U = self.U[:, dims]
            Qe = self.R[:, dims]
            V = self.V[:, dims]
            Kprime = self.K - U @ Q.T - Q @ U.T + Q @ G @ Q.T
            Lprime = self.L - V @ Q.T - Qe @ U.T + Qe @ G @ Q.T
            coef = np.linalg.solve(
                Kprime + RIDGE_ALPHA * np.eye(len(Kprime), dtype=np.float64),
                self.Yc,
            )
            return Lprime @ coef + self.ym

        Lcoef = self.Lc0 - self.LHF[:, take] @ w
        fcoef = rhs - self.FHF[np.ix_(take, take)] @ w
        ucoef, qcoef = fcoef[:d], fcoef[d:]
        Qe = self.R[:, dims]
        scores = Lcoef - self.V[:, dims] @ qcoef - Qe @ ucoef + Qe @ G @ qcoef
        return scores + self.ym

    def predict(self, dims: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        scores = self.scores(dims)
        return scores.argmax(1), scores


def ridge_fixed_direct(hfit: np.ndarray, yfit: np.ndarray, heval: np.ndarray,
                       classes: int, spec: dict[str, Any], dims: Sequence[int],
                       standardizer: tuple[np.ndarray, np.ndarray] | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Slow locked-protocol reference used by equivalence tests only."""
    if standardizer is None:
        mu = hfit.mean(0, dtype=np.float64)
        sd = hfit.std(0, dtype=np.float64)
        sd[sd < 1e-6] = 1.0
    else:
        mu, sd = standardizer
    hf = erase(hfit, spec, dims)
    he = erase(heval, spec, dims)
    X = ((hf - mu) / sd).astype(np.float32)
    Z = ((he - mu) / sd).astype(np.float32)
    Y = np.eye(classes, dtype=np.float64)[yfit]
    ym = Y.mean(0)
    Yc = Y - ym
    if X.shape[1] <= X.shape[0]:
        W = np.linalg.solve(X.T @ X + RIDGE_ALPHA * np.eye(X.shape[1]), X.T @ Yc)
        scores = Z @ W + ym
    else:
        K = (X @ X.T).astype(np.float64)
        coef = np.linalg.solve(K + RIDGE_ALPHA * np.eye(len(K)), Yc)
        scores = (Z @ X.T) @ coef + ym
    return scores.argmax(1), scores


def ce(labels:np.ndarray,scores:np.ndarray)->float:
    p=np.exp(scores-scores.max(1,keepdims=True));p/=p.sum(1,keepdims=True);return float(-np.log(np.clip(p[np.arange(len(labels)),labels],1e-12,1)).mean())


def bootstrap(values: Sequence[float], *seed_parts: object) -> tuple[float,float,float,float]:
    v=np.asarray(values,dtype=float); rng=np.random.default_rng(stable_seed(*seed_parts)); idx=rng.integers(0,len(v),size=(BOOTSTRAP_DRAWS,len(v))); means=v[idx].mean(1)
    return float(v.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975)),float(np.median(v))


def select_protected(h:np.ndarray,y:np.ndarray,sub:np.ndarray,sess:np.ndarray,spec:dict[str,Any],model:str,task:str,fold:int)->tuple[list[int],list[dict[str,Any]]]:
    subjects=natural_subjects(sub); rows=[]; selected=[]; all_dims=np.arange(spec["rank"])
    stats=[{"abs":{},"excess":{}} for _ in spec["blocks"]]
    qall=canonical(h,spec).astype(np.float32,copy=False);raw_base=_erasure_base(spec)
    # A split has one intact-train standardizer.  Reuse its kernel engine for
    # every protected block and all 100 matched random erasures.
    for split in range(UTILITY_SPLITS):
        vals=np.asarray(subjects,dtype=object); vals=vals[np.random.default_rng(stable_seed("half",model,task,fold,split)).permutation(len(vals))]; n=max(1,min(len(vals)-1,len(vals)//2)); fit=set(vals[:n]); eva=set(vals[n:])
        fi=np.flatnonzero(np.isin(sub,list(fit))); ei=np.flatnonzero(np.isin(sub,list(eva)))
        engine=FixedStandardizerKernelRidge(h[fi],y[fi],h[ei],spec.get("classes",int(y.max()+1)),spec,qfit=qall[fi],qeval=qall[ei],raw_base=raw_base)
        _p0,s0=engine.predict(())
        for bi,block in enumerate(spec["blocks"]):
            _,sp=engine.predict(block)
            candidates=np.setdiff1d(all_dims,np.asarray(block)); random_scores=[]
            for draw in range(RANDOM_ERASURES):
                pool=candidates if len(candidates)>=len(block) else all_dims; rd=np.random.default_rng(stable_seed("utility-random",model,task,fold,split,bi,draw)).choice(pool,len(block),replace=False)
                _,sr=engine.predict(rd);random_scores.append(sr)
            for s in natural_subjects(sub[ei]):
                local=np.flatnonzero(sub[ei].astype(str)==s); bce=ce(y[ei][local],s0[local]); ah=ce(y[ei][local],sp[local])-bce
                rhs=[ce(y[ei][local],sr[local])-bce for sr in random_scores]
                stats[bi]["abs"].setdefault(s,[]).append(ah);stats[bi]["excess"].setdefault(s,[]).append(ah-float(np.mean(rhs)))
        del engine;gc.collect()
    for bi,block in enumerate(spec["blocks"]):
        av=[float(np.mean(v)) for v in stats[bi]["abs"].values()];ev=[float(np.mean(v)) for v in stats[bi]["excess"].values()]
        am,alo,ahi,_=bootstrap(av,"assign-abs",model,task,fold,bi);em,elo,ehi,_=bootstrap(ev,"assign-excess",model,task,fold,bi)
        supported=bool(spec["support"][bi]["persistence_supported"]); protected=bool(supported and alo>0 and elo>0)
        rows.append({"Model":model,"Task":task,"fold":fold,"block":bi,"dimensions":len(block),"persistence_supported":supported,"absolute_CE_harm":am,"absolute_CI_low":alo,"absolute_CI_high":ahi,"excess_CE_harm":em,"excess_CI_low":elo,"excess_CI_high":ehi,"protected":protected})
        if protected:selected.extend(block)
    return sorted(set(selected)),rows


def evaluate(model:str,task:str,fold:int,device:torch.device)->dict[str,Any]:
    path=RUNTIME/"cells"/model.lower()/task.lower()/f"fold{fold}_seed0.json"
    if path.is_file(): return json.loads(path.read_text(encoding="utf-8"))
    row=next(x for x in all_checkpoint_rows() if x["Model"]==model and x["Task"]==task and x["fold"]==fold)
    data=load_fold_arrays(task,fold)
    if data["normalizer"]["mean_std_sha256"]!=row["normalizer_sha256"]: raise RuntimeError("normalizer mismatch")
    x,y,sub,sess,xe,ye,se=prepare_capped(data,task,fold)
    net,head=build_model(row,device); h=representations(net,head,x,model,device); he=representations(net,head,xe,model,device)
    if h.shape[1] != int(head.in_features): raise RuntimeError("representation/head mismatch")
    spec=spectrum(h,y,sub,sess,task,model,fold);spec["classes"]=data["classes"]
    protected,assignment=select_protected(h,y,sub,sess,spec,model,task,fold)
    fit=np.flatnonzero(sess==data["source_session"]); qfit=canonical(h[fit],spec); qe=canonical(he,spec); engine=FixedStandardizerKernelRidge(h[fit],y[fit],he,data["classes"],spec,qfit=qfit,qeval=qe,raw_base=_erasure_base(spec)); p0,s0=engine.predict(()); pp,sp=engine.predict(protected)
    all_dims=np.arange(spec["rank"]); random_predictions=[]
    for draw in range(RANDOM_ERASURES):
        rd=[] if not protected else np.random.default_rng(stable_seed("final-random",model,task,fold,draw)).choice(all_dims,len(protected),replace=False).tolist()
        pr,_=engine.predict(rd);random_predictions.append(pr)
    subjects=[]
    for s in natural_subjects(se):
        m=se.astype(str)==s; b0=float(balanced_accuracy_score(ye[m],p0[m]));bp=float(balanced_accuracy_score(ye[m],pp[m]));br=float(np.mean([balanced_accuracy_score(ye[m],p[m]) for p in random_predictions]));subjects.append({"subject_id":s,"intact_BA":b0,"protected_BA":bp,"random_BA":br,"protected_harm_pp":100*(b0-bp),"random_harm_pp":100*(b0-br),"PEEH_pp":100*(br-bp)})
    result={"Model":model,"Task":task,"fold":fold,"seed":0,"representation_dim":int(h.shape[1]),"trainable_parameters":int(row["trainable_parameters"]),"protected_dimensions":len(protected),"protected_blocks":protected,"protected_assignment":assignment,"subjects":subjects,"checkpoint_sha256":row["checkpoint_sha256"],"rank":spec["rank"],"numerical_rank_lower_bound":spec["numerical_rank_lower_bound"],"capped_train_rows":len(h),"capped_evaluation_rows":len(he),"svd_solver":spec["svd_solver"]}
    write_json(path,result);print(f"PEEH_CELL_COMPLETE {model} {task} fold={fold} protected_rank={len(protected)}",flush=True)
    del net,h,he,x,xe;gc.collect();torch.cuda.empty_cache() if device.type=="cuda" else None
    return result


def aggregate() -> None:
    cells=[json.loads((RUNTIME/"cells"/m.lower()/t.lower()/f"fold{f}_seed0.json").read_text()) for m in MODELS for t in TASKS for f in FOLDS]
    run_rows=[];subject_raw=[]
    for c in cells:
        vals=c["subjects"];run_rows.append({"Model":c["Model"],"Task":c["Task"],"fold":c["fold"],"seed":0,"Representation dim":c["representation_dim"],"trainable_parameters":c["trainable_parameters"],"protected_rank":c["protected_dimensions"],"Intact probe BA":np.mean([x["intact_BA"] for x in vals]),"Protected-erased BA":np.mean([x["protected_BA"] for x in vals]),"Random-erased BA":np.mean([x["random_BA"] for x in vals]),"Protected harm pp":np.mean([x["protected_harm_pp"] for x in vals]),"Random harm pp":np.mean([x["random_harm_pp"] for x in vals]),"PEEH pp":np.mean([x["PEEH_pp"] for x in vals])});subject_raw.extend([{"Model":c["Model"],"Task":c["Task"],"fold":c["fold"],**x} for x in vals])
    subject_rows=[]
    for model in MODELS:
        for task in TASKS:
            ids=natural_subjects([x["subject_id"] for x in subject_raw if x["Model"]==model and x["Task"]==task])
            for s in ids:
                z=[x for x in subject_raw if x["Model"]==model and x["Task"]==task and x["subject_id"]==s];subject_rows.append({"Model":model,"Task":task,"subject_id":s,**{k:float(np.mean([x[k] for x in z])) for k in ("intact_BA","protected_BA","random_BA","protected_harm_pp","random_harm_pp","PEEH_pp")},"folds":len(z)})
    primary=[]
    for model in MODELS:
        for task in TASKS:
            s=[x for x in subject_rows if x["Model"]==model and x["Task"]==task];r=[x for x in run_rows if x["Model"]==model and x["Task"]==task];mean,lo,hi,med=bootstrap([x["PEEH_pp"] for x in s],"final",model,task)
            primary.append({"Model":model,"Task":task,"Representation dim":r[0]["Representation dim"],"Intact probe BA":np.mean([x["intact_BA"] for x in s]),"Protected-erased BA":np.mean([x["protected_BA"] for x in s]),"Random-erased BA":np.mean([x["random_BA"] for x in s]),"Protected harm pp":np.mean([x["protected_harm_pp"] for x in s]),"Random harm pp":np.mean([x["random_harm_pp"] for x in s]),"PEEH pp":mean,"PEEH CI low":lo,"PEEH CI high":hi,"Median PEEH pp":med,"Protected assignment coverage":f"{sum(x['protected_rank']>0 for x in r)}/5","Significant consequence":"YES" if lo>0 else "NO"})
    summary=[]
    for model in MODELS:
        z=[x for x in primary if x["Model"]==model];summary.append({"Model":model,"MI significant?":next(x for x in z if x["Task"]=="OpenBMI_MI")["Significant consequence"],"ERP significant?":next(x for x in z if x["Task"]=="OpenBMI_ERP")["Significant consequence"],"SSVEP significant?":next(x for x in z if x["Task"]=="OpenBMI_SSVEP")["Significant consequence"],"WBCIC significant?":next(x for x in z if x["Task"]=="WBCIC_MI")["Significant consequence"],"Significant tasks / 4":sum(x["Significant consequence"]=="YES" for x in z),"Mean Protected assignment coverage":np.mean([int(x["Protected assignment coverage"].split('/')[0])/5 for x in z])})
    write_csv(OUT/"RUN_LEVEL_PEEH.csv",run_rows);write_csv(OUT/"SUBJECT_LEVEL_PEEH.csv",subject_rows);write_csv(OUT/"CROSSBACKBONE_PEEH_SEED0.csv",primary);write_csv(OUT/"CROSSBACKBONE_CONSEQUENCE_SUMMARY.csv",summary)
    reps=[]
    for model in MODELS:
        for task in TASKS:
            r=next(x for x in run_rows if x["Model"]==model and x["Task"]==task);reps.append({"Model":model,"Task":task,"Representation dim":r["Representation dim"],"trainable_parameters":r["trainable_parameters"]})
    write_csv(OUT/"REPRESENTATION_SPEC.csv",reps)
    lines=["# Frozen cross-backbone PEEH (seed 0)","","TFFormer was excluded by user instruction. SGN is deferred because 6/20 seed0 frozen checkpoints are absent and this analysis is forbidden to train.","","|Model|Task|Params|D|Intact BA|Protected BA|Random BA|PEEH pp|95% CI|Coverage|Significant|","|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for x in primary:
        p=next(r["trainable_parameters"] for r in reps if r["Model"]==x["Model"] and r["Task"]==x["Task"]);lines.append(f"|{x['Model']}|{x['Task']}|{p:,}|{x['Representation dim']:,}|{100*x['Intact probe BA']:.2f}|{100*x['Protected-erased BA']:.2f}|{100*x['Random-erased BA']:.2f}|{x['PEEH pp']:.3f}|[{x['PEEH CI low']:.3f}, {x['PEEH CI high']:.3f}]|{x['Protected assignment coverage']}|{x['Significant consequence']}|")
    lines += ["","Interpretation is restricted to persistence-consequence consistency. Absolute PEEH is not a model-quality ranking."]
    (OUT/"FINAL_CROSSBACKBONE_PEEH_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("CROSSBACKBONE_PEEH_100_RUNS_COMPLETE",flush=True)


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("prelock","run","aggregate"),required=True);p.add_argument("--model",choices=MODELS);p.add_argument("--task",choices=TASKS);p.add_argument("--fold",type=int,choices=FOLDS);a=p.parse_args()
    if a.stage=="prelock":prelock()
    elif a.stage=="aggregate":aggregate()
    else:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        jobs=[(m,t,f) for m in MODELS for t in TASKS for f in FOLDS if (a.model is None or m==a.model) and (a.task is None or t==a.task) and (a.fold is None or f==a.fold)]
        for m,t,f in jobs:evaluate(m,t,f,device)
    return 0


if __name__=="__main__":raise SystemExit(main())
