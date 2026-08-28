from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
PROTOCOL = EXP / "protocol"
RUNTIME = EXP / "runtime"
FM_RUNTIME = Path(r"D:\nips-temp\TotalP\P2\fm_rescue_runtime")
PYTHON = Path(r"D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe")

FMS = ("CBraMod", "LaBraM")
DATASETS = ("OpenBMI", "WBCIC")
FOLDS = tuple(range(5))
SEEDS = (0, 1, 2)
SOURCE_SESSIONS = {"OpenBMI": (1, 2), "WBCIC": (0, 1)}
FUTURE_SESSION = {"OpenBMI": None, "WBCIC": 2}
LR_GRIDS = {"CBraMod": (1e-4, 3e-4), "LaBraM": (1e-4, 5e-4)}
WEIGHT_DECAY = 0.05
MAX_EPOCHS = 12
MIN_EPOCHS = 4
PATIENCE = 3
BATCH_SIZE = 128
LABEL_SMOOTHING = 0.1
COMPETENCE_THRESHOLDS = {"OpenBMI": 0.7519166667, "WBCIC": 0.7684300821}
SPECIALIST_ANCHORS = {"OpenBMI": 0.7719166667, "WBCIC": 0.7884300821}

OPENBMI_CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10", "FC3", "FC4",
    "C5", "C1", "C2", "C6", "CP3", "CPz", "CP4", "P1", "P2", "POz", "FT9",
    "FTT9h", "TTP7h", "TP7", "TPP9h", "FT10", "FTT10h", "TPP8h", "TP8", "TPP10h",
    "F9", "F10", "AF7", "AF3", "AF4", "AF8", "PO3", "PO4",
)
WBCIC_CHANNELS = (
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "Fz", "F1", "F2", "F3", "F4",
    "F5", "F6", "F7", "F8", "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7",
    "FT8", "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8", "CP1", "CP2",
    "CP3", "CP4", "CP5", "CP6", "TP7", "TP8", "P3", "P4", "P5", "P6", "P7", "P8",
    "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "Oz", "O1", "O2",
)
STANDARD_1020 = (
    "FP1", "FPZ", "FP2", "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10", "FT9", "FT7", "FC5", "FC3",
    "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10", "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "T10", "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10", "P9",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10", "PO9", "PO7", "PO5", "PO3", "PO1",
    "POZ", "PO2", "PO4", "PO6", "PO8", "PO10", "O1", "OZ", "O2", "O9", "CB1", "CB2", "IZ", "O10", "T3",
    "T5", "T4", "T6", "M1", "M2", "A1", "A2", "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7",
    "CFC8", "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8", "T1", "T2", "FTT9H", "TTP7H",
    "TPP9H", "FTT10H", "TPP8H", "TPP10H",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P2 = _load_module("fm_rescue_p2", REPO / "experiments" / "persist_eeg_subject_invariance_stress_test_v1" / "code" / "common.py")
P3 = _load_module("fm_rescue_p3", REPO / "experiments" / "persist_eeg_wbcic_independent_replication_v1" / "code" / "common.py")


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def ensure_dirs() -> None:
    for path in (RESULTS, FIGURES, PROTOCOL, RUNTIME, RUNTIME / "anchors", RUNTIME / "representations"):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    x = np.ascontiguousarray(value)
    h = hashlib.sha256(str(x.dtype).encode() + np.asarray(x.shape, np.int64).tobytes() + x.tobytes())
    return h.hexdigest()


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed) % (2**32 - 1)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def subject_sort(values: Iterable[str]) -> list[str]:
    return sorted((str(v).replace("sub-", "") for v in values), key=lambda x: (int(x) if x.isdigit() else 10**9, x))


@dataclass
class DataBundle:
    dataset: str
    raw: np.ndarray
    metadata: pd.DataFrame
    channels: tuple[str, ...]
    cache_root: Path


def load_data(dataset: str) -> DataBundle:
    if dataset == "OpenBMI":
        d = P2.load_data(); meta = d.metadata.copy(); channels = OPENBMI_CHANNELS
    elif dataset == "WBCIC":
        d = P3.load_data(); meta = d.metadata.copy(); channels = WBCIC_CHANNELS
    else:
        raise KeyError(dataset)
    meta["subject_id"] = meta.subject_id.astype(str).str.replace("sub-", "", regex=False)
    meta["session_id"] = meta.session_id.astype(int); meta["label"] = meta.label.astype(int)
    if d.x.shape[1] != len(channels) or d.x.shape[2] != 1000:
        raise RuntimeError(f"unexpected {dataset} cache shape {d.x.shape}")
    return DataBundle(dataset, d.x, meta.reset_index(drop=True), channels, Path(d.cache_root))


def fold_roles(dataset: str, fold: int) -> dict[str, tuple[str, ...]]:
    row = P2.frozen_fold(fold) if dataset == "OpenBMI" else P3.frozen_fold(fold)
    if dataset == "OpenBMI":
        result = {"model_fit": row["inner_train"], "validation": row["inner_validation"], "outcome": row["outcome"]}
    else:
        result = {"model_fit": row["model_fit"], "validation": row["validation_discovery"], "outcome": row["outcome"]}
    return {k: tuple(subject_sort(v)) for k, v in result.items()}


def row_indices(meta: pd.DataFrame, subjects: Sequence[str], sessions: Sequence[int]) -> np.ndarray:
    mask = meta.subject_id.astype(str).isin(set(map(str, subjects))).to_numpy(copy=True)
    mask &= meta.session_id.astype(int).isin(set(map(int, sessions))).to_numpy()
    return np.flatnonzero(mask).astype(np.int64)


def channels(dataset: str) -> tuple[str, ...]:
    return OPENBMI_CHANNELS if dataset == "OpenBMI" else WBCIC_CHANNELS


def labram_input_chans(dataset: str) -> list[int]:
    upper = [x.upper() for x in channels(dataset)]
    missing = [x for x in upper if x not in STANDARD_1020]
    if missing:
        raise RuntimeError(f"LaBraM channel vocabulary miss: {missing}")
    return [0] + [STANDARD_1020.index(x) + 1 for x in upper]


def preprocessed_path(dataset: str) -> Path:
    return RUNTIME / f"{dataset.upper()}_FM_INPUT_UV_200HZ.npy"


def preprocessed_mask_path(dataset: str) -> Path:
    return RUNTIME / f"{dataset.upper()}_FM_INPUT_MASK.npy"


def prepare_inputs(dataset: str, include_future: bool = False, chunk: int = 24) -> dict[str, Any]:
    ensure_dirs(); data = load_data(dataset); target = preprocessed_path(dataset); mask_path = preprocessed_mask_path(dataset)
    shape = (len(data.metadata), len(data.channels), 800)
    if target.is_file(): out = np.lib.format.open_memmap(target, mode="r+")
    else: out = np.lib.format.open_memmap(target, mode="w+", dtype=np.float16, shape=shape)
    if mask_path.is_file(): done = np.load(mask_path, allow_pickle=False)
    else: done = np.zeros(len(data.metadata), dtype=bool)
    allowed_sessions = set(SOURCE_SESSIONS[dataset])
    if include_future and FUTURE_SESSION[dataset] is not None: allowed_sessions.add(int(FUTURE_SESSION[dataset]))
    pending = np.flatnonzero(data.metadata.session_id.astype(int).isin(allowed_sessions).to_numpy() & ~done)
    sos = butter(6, 40.0, btype="lowpass", fs=250.0, output="sos") if dataset == "OpenBMI" else None
    for start in range(0, len(pending), chunk):
        idx = pending[start:start + chunk]
        x = np.asarray(data.raw[idx], dtype=np.float32)
        x *= (1e6 if dataset == "OpenBMI" else 20.0)
        if sos is not None: x = sosfiltfilt(sos, x, axis=-1).astype(np.float32)
        x = resample_poly(x, 4, 5, axis=-1).astype(np.float32)
        if x.shape[-1] != 800: raise RuntimeError(x.shape)
        out[idx] = x.astype(np.float16); done[idx] = True
        if (start // chunk) % 20 == 0:
            out.flush(); np.save(mask_path, done, allow_pickle=False)
            print(f"[prepare] {dataset} {min(start+len(idx),len(pending))}/{len(pending)}", flush=True)
    out.flush(); np.save(mask_path, done, allow_pickle=False)
    selected = np.flatnonzero(done)
    sample = np.asarray(out[selected[: min(len(selected), 256)]], dtype=np.float32)
    audit = {"dataset": dataset, "shape": list(shape), "processed_rows": int(done.sum()), "future_included": bool(include_future),
             "sampling_rate_hz": 200, "unit": "microvolts", "finite": bool(np.isfinite(sample).all()),
             "sample_abs_median": float(np.median(np.abs(sample))), "sample_abs_q99": float(np.quantile(np.abs(sample), .99)),
             "input_path": str(target), "mask_sha256": sha256(mask_path)}
    write_json(RUNTIME / f"{dataset.upper()}_FM_INPUT_RUNTIME_AUDIT.json", audit)
    return audit


class FMTask(nn.Module):
    def __init__(self, fm: str, dataset: str, seed: int):
        super().__init__(); self.fm = fm; self.dataset = dataset; set_seed(seed)
        if fm == "CBraMod":
            root = FM_RUNTIME / "CBraMod"; sys.path.insert(0, str(root))
            from models.cbramod import CBraMod
            self.encoder = CBraMod(); payload = torch.load(root / "pretrained_weights" / "pretrained_weights.pth", map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(payload, strict=True); self.encoder.proj_out = nn.Identity(); self.head = nn.Linear(200, 2)
            nn.init.trunc_normal_(self.head.weight, std=.02); nn.init.zeros_(self.head.bias); self.input_chans = None
        elif fm == "LaBraM":
            root = FM_RUNTIME / "LaBraM"; sys.path.insert(0, str(root)); import modeling_finetune
            self.encoder = modeling_finetune.labram_base_patch200_200(pretrained=False, num_classes=0, use_mean_pooling=True,
                use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1, qkv_bias=False)
            payload = torch.load(root / "checkpoints" / "labram-base.pth", map_location="cpu", weights_only=False)
            state = payload.get("model", payload); cleaned = {}
            for key, value in state.items():
                key = key.removeprefix("student.")
                if not key.startswith("head."): cleaned[key] = value
            self.encoder.load_state_dict(cleaned, strict=False); self.head = nn.Linear(200, 2)
            nn.init.trunc_normal_(self.head.weight, std=.02); nn.init.zeros_(self.head.bias)
            self.input_chans = labram_input_chans(dataset)
        else: raise KeyError(fm)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.fm == "CBraMod":
            tokens = self.encoder(x); return tokens.mean(dim=(1, 2))
        return self.encoder.forward_features(x / 100.0, input_chans=self.input_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def model_checkpoint(fm: str, dataset: str, fold: int, seed: int) -> Path:
    return RUNTIME / "anchors" / fm / dataset / f"fold-{fold}" / f"seed-{seed}.pt"


def search_checkpoint(fm: str, dataset: str, fold: int, lr: float) -> Path:
    return RUNTIME / "search" / fm / dataset / f"fold-{fold}" / f"lr-{lr:.0e}.pt"


def mean_subject_ba(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> float:
    pred = logits.argmax(1); values = []
    for subject in subject_sort(np.unique(subjects.astype(str))):
        m = subjects.astype(str) == subject; values.append(balanced_accuracy_score(labels[m], pred[m]))
    return float(np.mean(values))


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    pred = logits.argmax(1); z = logits.astype(np.float64); z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return {"BA": float(balanced_accuracy_score(labels, pred)), "macro_F1": float(f1_score(labels, pred, average="macro", zero_division=0)),
            "NLL": float(log_loss(labels, p, labels=[0, 1]))}


def input_array(dataset: str) -> np.ndarray:
    path = preprocessed_path(dataset)
    if not path.is_file(): raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


@torch.no_grad()
def infer(model: FMTask, dataset: str, indices: np.ndarray, device: torch.device, representations: bool = True) -> dict[str, np.ndarray]:
    model.eval(); xall = input_array(dataset); data = load_data(dataset); logits=[]; feats=[]
    for start in range(0, len(indices), BATCH_SIZE):
        idx = indices[start:start+BATCH_SIZE]; x = torch.from_numpy(np.asarray(xall[idx], dtype=np.float32)).to(device, non_blocking=True).reshape(len(idx), len(channels(dataset)), 4, 200)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h = model.forward_features(x); z = model.head(h)
        logits.append(z.float().cpu().numpy());
        if representations: feats.append(h.float().cpu().numpy())
    selected = data.metadata.iloc[indices]
    return {"indices": indices.copy(), "features": np.concatenate(feats).astype(np.float32) if representations else np.empty((len(indices),0),np.float32),
            "logits": np.concatenate(logits).astype(np.float32), "labels": selected.label.to_numpy(np.int64),
            "subjects": selected.subject_id.astype(str).to_numpy(), "sessions": selected.session_id.to_numpy(np.int64)}


def train_anchor(fm: str, dataset: str, fold: int, seed: int, lr: float, output: Path, device: torch.device) -> dict[str, Any]:
    if output.is_file():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        if payload.get("complete") is True: return payload["record"]
    data = load_data(dataset); roles = fold_roles(dataset, fold); sessions = SOURCE_SESSIONS[dataset]
    train_idx = row_indices(data.metadata, roles["model_fit"], sessions); val_idx = row_indices(data.metadata, roles["validation"], sessions)
    done = np.load(preprocessed_mask_path(dataset), allow_pickle=False)
    if not done[train_idx].all() or not done[val_idx].all(): raise RuntimeError("source-validation preprocessing incomplete")
    set_seed(stable_seed("fm-anchor", fm, dataset, fold, seed)); model = FMTask(fm, dataset, stable_seed("head", fm, dataset, fold, seed)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, betas=(.9,.999)); criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    xall = input_array(dataset); labels = data.metadata.label.to_numpy(np.int64); best = -np.inf; best_epoch=-1; best_state=None; stale=0; history=[]
    rng = np.random.default_rng(stable_seed("order", fm, dataset, fold, seed));
    for epoch in range(MAX_EPOCHS):
        model.train(); order=train_idx.copy(); rng.shuffle(order); losses=[]
        warm = min(1.0, (epoch+1)/2.0); cosine = .5*(1+math.cos(math.pi*epoch/MAX_EPOCHS)); now_lr=lr*warm*(.1+.9*cosine)
        for group in opt.param_groups: group["lr"] = now_lr
        for start in range(0,len(order),BATCH_SIZE):
            idx=order[start:start+BATCH_SIZE]; x=torch.from_numpy(np.asarray(xall[idx],dtype=np.float32)).to(device).reshape(len(idx),len(channels(dataset)),4,200); y=torch.as_tensor(labels[idx],device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"): z=model(x); loss=criterion(z,y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step(); losses.append(float(loss.detach()))
        val=infer(model,dataset,val_idx,device,representations=False); score=mean_subject_ba(val["labels"],val["logits"],val["subjects"])
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"validation_mean_subject_BA":score,"lr":now_lr})
        if score > best + 1e-6:
            best=score; best_epoch=epoch+1; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}; stale=0
        else: stale += 1
        print(f"[train] {fm} {dataset} fold={fold} seed={seed} lr={lr:g} epoch={epoch+1} valBA={score:.5f} best={best:.5f}",flush=True)
        if epoch+1 >= MIN_EPOCHS and stale >= PATIENCE: break
    record={"fm":fm,"dataset":dataset,"fold":fold,"seed":seed,"lr":lr,"weight_decay":WEIGHT_DECAY,"best_epoch":best_epoch,
            "validation_mean_subject_BA":float(best),"train_rows":len(train_idx),"validation_rows":len(val_idx),"history":history,
            "target_seen_by_anchor":False,"future_session_used":False}
    output.parent.mkdir(parents=True,exist_ok=True); temp=output.with_suffix(".pt.part"); torch.save({"complete":True,"state_dict":best_state,"record":record},temp); os.replace(temp,output)
    del model; torch.cuda.empty_cache(); return record


def load_anchor(fm: str, dataset: str, fold: int, seed: int, device: torch.device) -> FMTask:
    path=model_checkpoint(fm,dataset,fold,seed); payload=torch.load(path,map_location="cpu",weights_only=False); model=FMTask(fm,dataset,stable_seed("head",fm,dataset,fold,seed)); model.load_state_dict(payload["state_dict"],strict=True); return model.eval().to(device)


def chronological_class_split(labels: np.ndarray, fraction: float=.7) -> tuple[np.ndarray,np.ndarray]:
    train=[]; val=[]
    for label in sorted(np.unique(labels)):
        p=np.flatnonzero(labels==label); cut=min(max(int(np.floor(len(p)*fraction)),1),len(p)-1); train.extend(p[:cut]); val.extend(p[cut:])
    return np.asarray(sorted(train),np.int64),np.asarray(sorted(val),np.int64)


def adapt_linear_head(features: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
                      weight: np.ndarray, bias: np.ndarray, lr: float, seed: int, max_epochs: int=50) -> dict[str,Any]:
    set_seed(seed); device=torch.device("cuda"); h=torch.from_numpy(features).float().to(device); y=torch.from_numpy(labels).long().to(device)
    head=nn.Linear(features.shape[1],2).to(device); head.weight.data.copy_(torch.from_numpy(weight).to(device)); head.bias.data.copy_(torch.from_numpy(bias).to(device)); initial=np.concatenate([weight.ravel(),bias.ravel()])
    opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=1e-4); best=-np.inf; state=None; epoch_best=0; stale=0
    for epoch in range(max_epochs):
        head.train(); opt.zero_grad(set_to_none=True); loss=nn.functional.cross_entropy(head(h[train_idx]),y[train_idx]); loss.backward(); opt.step(); head.eval()
        with torch.no_grad(): z=head(h); score=balanced_accuracy_score(labels[val_idx],z[val_idx].argmax(1).cpu().numpy())
        if score>best+1e-8: best=score; state={k:v.detach().clone() for k,v in head.state_dict().items()}; epoch_best=epoch+1; stale=0
        else: stale+=1
        if epoch+1>=10 and stale>=8: break
    head.load_state_dict(state); head.eval(); logits=head(h).detach().cpu().numpy().astype(np.float32); final=np.concatenate([head.weight.detach().cpu().numpy().ravel(),head.bias.detach().cpu().numpy().ravel()])
    return {"logits":logits,"weight":head.weight.detach().cpu().numpy(),"bias":head.bias.detach().cpu().numpy(),"best_epoch":epoch_best,
            "parameter_relative_change":float(np.linalg.norm(final-initial)/max(np.linalg.norm(initial),1e-12))}
