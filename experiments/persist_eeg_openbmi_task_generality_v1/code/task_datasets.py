"""Locked task-cache access, splits, model construction, and deterministic training primitives."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

REPO = Path(os.environ.get("TASK_GENERALITY_REPO", Path(__file__).resolve().parents[3])).resolve()
EXP = REPO / "experiments" / "persist_eeg_openbmi_task_generality_v1"
PROTOCOL = EXP / "protocol"
OUTPUTS = EXP / "outputs"
RUNTIME = Path(os.environ.get("TASK_GENERALITY_RUNTIME", "/root/rivermind-data/openbmi_task_generality_runtime")).resolve()
CACHE_ROOT = Path(os.environ.get("PERSIST_OPENBMI_CACHE", "/root/rivermind-data/persist_eeg_cache/openbmi/openbmi")).resolve()
FIVEFOLD = REPO / "experiments" / "persist_eeg_carrier_5fold_multiseed_stability_v1" / "protocol" / "FIVEFOLD_SPLIT.json"
FINAL_HOLDOUT = REPO / "experiments" / "persist_eeg_final_heldout_confirmation_v1" / "protocol" / "FINAL_HOLDOUT_MANIFEST.json"
V8_SPLIT = REPO / "experiments" / "persist_eeg_final_model_v8" / "outputs" / "protocol" / "V8_SEARCH_SPLIT.json"
CARRIER_CODE = REPO / "experiments" / "persist_eeg_carrier_dualdataset_screen_v1" / "code"

TASKS = {
    "ERP": {"cache_name": "erp", "classes": 2, "samples": 250, "label_map": {1: 0, 2: 1}, "source_session": 1, "future_session": 2, "weighted_ce": True},
    "SSVEP": {"cache_name": "ssvep", "classes": 4, "samples": 1000, "label_map": {1: 0, 2: 1, 3: 2, 4: 3}, "source_session": 1, "future_session": 2, "weighted_ce": False},
}
SEEDS = (0, 1, 2)
EPOCHS, MIN_EPOCH, BATCH_SIZE = 60, 10, 64
LR, WEIGHT_DECAY, CLIP = 3e-4, 5e-4, 5.0


def clean(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def sha_bytes(value: bytes) -> str: return hashlib.sha256(value).hexdigest()


def sort_subjects(values: Iterable[str]) -> list[str]: return sorted(map(str, values), key=lambda x:int(x.replace("sub-", "")))


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True


def state_hash(model: nn.Module) -> str:
    import io
    b=io.BytesIO(); torch.save(model.state_dict(),b); return sha_bytes(b.getvalue())


def task_path(task: str, subject: str, session: int, kind: str) -> Path:
    spec=TASKS[task]; base=CACHE_ROOT/f"sub-{int(subject):02d}"/f"ses-{session}"/f"{spec['cache_name']}_1train"
    return base.with_name(base.name + ("_signals.npy" if kind=="signal" else "_codes.npy"))


@dataclass(frozen=True)
class Row:
    subject: str
    session: int
    signal_path: str
    index: int
    label: int


class TaskBundle:
    def __init__(self, task: str, subjects: list[str], rows: list[Row]):
        self.task, self.subjects, self.rows = task, sort_subjects(subjects), rows
        self.channels, self.samples = 62, TASKS[task]["samples"]
        self._arrays: dict[str,np.ndarray] = {}

    def indices(self, subjects: Iterable[str], sessions: Iterable[int]) -> np.ndarray:
        ss, se=set(map(str,subjects)),set(map(int,sessions))
        return np.asarray([i for i,row in enumerate(self.rows) if row.subject in ss and row.session in se],dtype=np.int64)

    def labels(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray([self.rows[int(i)].label for i in indices],dtype=np.int64)

    def signal_batch(self, indices: np.ndarray) -> np.ndarray:
        out=[]
        for i in np.asarray(indices,dtype=np.int64):
            row=self.rows[int(i)]
            if row.signal_path not in self._arrays:
                self._arrays[row.signal_path]=np.load(row.signal_path,mmap_mode="r",allow_pickle=False)
            out.append(np.asarray(self._arrays[row.signal_path][row.index],dtype=np.float32))
        return np.stack(out,axis=0)


def load_bundle(task: str, subjects: list[str], sessions: tuple[int, ...] = (1, 2)) -> TaskBundle:
    if task not in TASKS: raise ValueError(task)
    spec=TASKS[task]; rows=[]
    for subject in sort_subjects(subjects):
        for session in sessions:
            xp,yp=task_path(task,subject,session,"signal"),task_path(task,subject,session,"label")
            if not xp.is_file() or not yp.is_file(): raise FileNotFoundError(f"task cache missing: {task} sub-{subject} ses-{session}")
            x=np.load(xp,mmap_mode="r",allow_pickle=False); y=np.load(yp,mmap_mode="r",allow_pickle=False)
            if x.ndim!=3 or x.shape[1:]!=(62,spec["samples"]) or x.dtype!=np.float32 or y.shape!=(x.shape[0],):
                raise RuntimeError(f"cache schema mismatch: {xp}")
            raw=set(map(int,np.unique(y)))
            if raw != set(spec["label_map"]): raise RuntimeError(f"label schema mismatch: {yp}: {raw}")
            rows.extend(Row(subject,session,str(xp),i,spec["label_map"][int(code)]) for i,code in enumerate(y))
    return TaskBundle(task,subjects,rows)


def split_reference() -> tuple[list[str], list[str], dict[str,Any], str]:
    fold_data=json.loads(FIVEFOLD.read_text(encoding="utf-8"))
    folds=fold_data["folds"]["OpenBMI"]
    search=sort_subjects(fold_data["search_subjects"]["OpenBMI"])
    holdout=sort_subjects(json.loads(FINAL_HOLDOUT.read_text(encoding="utf-8"))["OpenBMI"]["subject_ids"])
    v8=sort_subjects(json.loads(V8_SPLIT.read_text(encoding="utf-8"))["openbmi"]["V8_SEARCH"])
    if len(search)!=40 or len(holdout)!=14 or search!=v8 or set(search)&set(holdout): raise RuntimeError("MI split membership mismatch")
    if len(folds)!=5: raise RuntimeError("five folds required")
    for f in folds:
        pieces=[set(map(str,f[k])) for k in ("inner_train_subjects","inner_val_subjects","outer_dev_subjects")]
        if any(pieces[a]&pieces[b] for a in range(3) for b in range(a+1,3)) or set.union(*pieces)!=set(search): raise RuntimeError("invalid frozen fold")
    return search,holdout,{"source":str(FIVEFOLD),"source_sha256":sha256(FIVEFOLD),"folds":folds},sha256(FINAL_HOLDOUT)


def normalizer(bundle: TaskBundle, subjects: list[str]) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    idx=bundle.indices(subjects,(1,)); total=np.zeros(62,np.float64); sq=np.zeros(62,np.float64); n=0
    for start in range(0,len(idx),64):
        x=bundle.signal_batch(idx[start:start+64]).astype(np.float64)
        total+=x.sum((0,2)); sq+=np.square(x).sum((0,2)); n+=x.shape[0]*x.shape[2]
    mean=(total/n).astype(np.float32); std=np.sqrt(np.maximum(sq/n-mean.astype(np.float64)**2,1e-12)).astype(np.float32)
    return mean,std,{"subjects":sort_subjects(subjects),"sessions":[1],"trials":int(len(idx)),"samples_per_channel":int(n),"mean_std_sha256":sha_bytes(mean.tobytes()+std.tobytes())}


def class_weights(bundle: TaskBundle, subjects: list[str]) -> tuple[torch.Tensor|None,dict[str,Any]]:
    y=bundle.labels(bundle.indices(subjects,(1,))); classes=TASKS[bundle.task]["classes"]; counts=np.bincount(y,minlength=classes)
    if np.any(counts==0): raise RuntimeError("inner train lacks a class")
    weights=(len(y)/(classes*counts)).astype(np.float32) if TASKS[bundle.task]["weighted_ce"] else None
    return (torch.from_numpy(weights) if weights is not None else None),{"counts":counts.tolist(),"weighted_cross_entropy":weights is not None,"weights":None if weights is None else weights.tolist(),"formula":"N/(K*N_k)" if weights is not None else "ordinary multiclass cross entropy"}


class RawGPUCache:
    """Materializes only the active SEARCH bundle once; normalizers remain fold-specific."""
    def __init__(self,bundle:TaskBundle,device:torch.device):
        xs=[]
        for start in range(0,len(bundle.rows),128): xs.append(torch.from_numpy(bundle.signal_batch(np.arange(start,min(start+128,len(bundle.rows)),dtype=np.int64))))
        self.x=torch.cat(xs,0).to(device,non_blocking=True); self.y=torch.as_tensor(bundle.labels(np.arange(len(bundle.rows),dtype=np.int64)),device=device,dtype=torch.long); self.device=device
    def batch(self,indices:np.ndarray,mean:np.ndarray,std:np.ndarray)->tuple[torch.Tensor,torch.Tensor]:
        ii=torch.as_tensor(indices,device=self.device,dtype=torch.long); x=self.x.index_select(0,ii)
        m=torch.as_tensor(mean,device=self.device)[None,:,None]; s=torch.as_tensor(std,device=self.device)[None,:,None]
        return (x-m)/torch.clamp(s,min=1e-6),self.y.index_select(0,ii)


def epoch_batches(indices: np.ndarray, fold_id: int, task: str, epoch: int) -> list[np.ndarray]:
    task_code=17 if task=="ERP" else 29
    rng=np.random.default_rng(1000003*(epoch+1)+1009*fold_id+task_code)
    shuffled=rng.permutation(indices)
    return [shuffled[start:start+BATCH_SIZE] for start in range(0,len(shuffled),BATCH_SIZE)]


def models() -> tuple[type[nn.Module],type[nn.Module]]:
    if str(CARRIER_CODE) not in sys.path: sys.path.insert(0,str(CARRIER_CODE))
    EEGNet=importlib.import_module("eegnet_locked").EEGNet
    CompactLite=importlib.import_module("run_carrier_screen").CompactLite
    return EEGNet,CompactLite


def build_model(name:str,task:str)->nn.Module:
    EEGNet,CompactLite=models(); spec=TASKS[task]
    model=EEGNet(62,samples=spec["samples"]) if name=="EEGNet" else CompactLite(62,"bn")
    model.head=nn.Linear(64,spec["classes"])
    return model


def verify_model(task:str)->dict[str,Any]:
    rows=[]
    for name in ("EEGNet","LiteBN"):
        model=build_model(name,task); x=torch.zeros((2,62,TASKS[task]["samples"])); logits,hidden=model(x)
        if logits.shape!=(2,TASKS[task]["classes"]) or hidden.shape!=(2,64): raise RuntimeError("task model output mismatch")
        rows.append({"model":name,"logits_shape":list(logits.shape),"hidden_shape":list(hidden.shape),"parameters":sum(p.numel() for p in model.parameters())})
    return {"task":task,"models":rows,"only_task_required_change":"classifier output dimension; EEGNet classifier input length follows canonical samples argument"}
