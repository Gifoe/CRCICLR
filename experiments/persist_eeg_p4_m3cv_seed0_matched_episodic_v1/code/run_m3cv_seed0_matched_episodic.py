"""M3CV / NEMAR nm000166 external matched-episodic replication, seed 0 only.

This runner implements the post-audit, frozen 93-subject protocol.  It reads
all valid native 4-s epochs for eligible people, imports final model classes
and the final episode sampler, and has no paths for other seeds or experiments.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import mne
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path(os.environ.get("M3CV_P4_ROOT", "/root/p4_m3cv_seed0_matched_episodic_v1")).resolve()
RAW_ROOT = Path(os.environ.get("M3CV_RAW_ROOT", "/root/m3cv_nm000166_external_replication")).resolve()
DATA = RAW_ROOT / "data" / "m3cv_nm000166"
CACHE_ROOT = RAW_ROOT / "cache" / "m3cv_lr_4s_250hz"
FINAL_REPO = Path(os.environ.get("SIRE_FINAL_REPO", "/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK")).resolve()

SEED, TRAINING_SEED = 0, 100_000
EPOCHS, MIN_EPOCH, EARLY_STOP_PATIENCE, EPISODE_TRIALS = 60, 10, 8, 128
LR, WEIGHT_DECAY, GRADIENT_CLIP = 3e-4, 5e-4, 5.0
C, T, FS = 64, 1000, 250
SESSIONS = (("ses-01", 1, "S1"), ("ses-02", 2, "S2"))
TASKS = (("motorLHand", 0, "left_hand"), ("motorRHand", 1, "right_hand"))

SIRE_SOURCE = FINAL_REPO / "experiments/persist_eeg_carrier_dualdataset_screen_v1/code/run_carrier_screen.py"
EEGNET_SOURCE = FINAL_REPO / "experiments/persist_eeg_carrier_dualdataset_screen_v1/code/eegnet_locked.py"
EPISODE_SOURCE = FINAL_REPO / "experiments/persist_eeg_r2eeg_stage1_v1/code/run_stage1.py"
CORE_SOURCE = FINAL_REPO / "experiments/persist_eeg_r2eeg_stage1_v1/code/stage1_core.py"
TRAINING_CONFIG = FINAL_REPO / "experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/protocol/TRAINING_PROTOCOL.json"
TRAINING_SOURCE = FINAL_REPO / "experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/code/train_grid.py"
EXPECTED_SHA256 = {
    SIRE_SOURCE: "920af131aabc272317da128f42be9961d5592619ce85ca99192029d1181f126f",
    EEGNET_SOURCE: "f7c513c3f3cd1f326a74b4e419bd693e15ee8980a7c378f1c0bee8215b8b89dd",
    EPISODE_SOURCE: "40cd24d90a1919db14d9d7b5a0b976bc5e4943e82e8a9fec10bbb82d70d78b75",
    CORE_SOURCE: "5277ec7055974c953acf32df3fe58a05761e0439b66c43eca3049563940c21ce",
    TRAINING_CONFIG: "e03ba36815df3d8f808e558211a2cba739b7fc53c0179fc9cc8c6cab31adf936",
    TRAINING_SOURCE: "23f67f2c6ee75ce0babaab56a9e37b3210fa7b589d0347a417d1437fb1290c61",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)): return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows: raise RuntimeError(f"empty CSV prohibited: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows([clean(row) for row in rows])
    os.replace(temp, path)


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temp, path)


def set_seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def rng_state() -> dict[str, Any]:
    result: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(value: dict[str, Any]) -> None:
    random.setstate(value["python"]); np.random.set_state(value["numpy"]); torch.set_rng_state(value["torch"])
    if "cuda" in value and torch.cuda.is_available(): torch.cuda.set_rng_state_all(value["cuda"])


def state_hash(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO(); torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def subject_key(value: str) -> int: return int(str(value).replace("sub-", ""))


def direct_import(name: str, path: Path, prepend: list[Path]) -> Any:
    paths = [str(value) for value in prepend]; sys.path[:0] = paths
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
        return module
    finally:
        for value in paths:
            if value in sys.path: sys.path.remove(value)


def load_authoritative() -> tuple[Any, Any, Any, dict[str, Any]]:
    for path, expected in EXPECTED_SHA256.items():
        actual = sha256(path) if path.is_file() else None
        if actual != expected: raise RuntimeError(f"authoritative hash drift: {path}; expected {expected}, got {actual}")
    recipe = json.loads(TRAINING_CONFIG.read_text(encoding="utf-8"))
    required = {"optimizer":"AdamW", "lr":LR, "weight_decay":WEIGHT_DECAY, "episode_batch_size":EPISODE_TRIALS, "gradient_clipping":GRADIENT_CLIP, "epochs":EPOCHS, "scheduler":"none", "loss":"ordinary cross entropy", "checkpoint_selection":"inner future-session mean-subject BA, eligible epochs 10..60, earliest tie"}
    for key, expected in required.items():
        if recipe.get(key) != expected: raise RuntimeError(f"training recipe drift {key}: {recipe.get(key)!r}")
    previous = os.environ.get("R2EEG_REPO"); os.environ["R2EEG_REPO"] = str(FINAL_REPO)
    try:
        stage1 = direct_import("run_stage1", EPISODE_SOURCE, [EPISODE_SOURCE.parent])
        eegnet = direct_import("m3cv_final_eegnet", EEGNET_SOURCE, [EEGNET_SOURCE.parent])
        sire = direct_import("m3cv_final_sire", SIRE_SOURCE, [SIRE_SOURCE.parent])
    finally:
        if previous is None: os.environ.pop("R2EEG_REPO", None)
        else: os.environ["R2EEG_REPO"] = previous
    return stage1, eegnet, sire, required


def audit_models(eegnet_module: Any, sire_module: Any) -> dict[str, Any]:
    models = {"EEGNet": eegnet_module.EEGNet(C, T), "SIRE-EEG": sire_module.CompactLite(C, "bn")}
    expected = {
        "EEGNet": {"temporal.weight":(8,1,1,64), "spatial.weight":(16,1,C,1), "depth.weight":(16,1,1,16), "point.weight":(16,16,1,1), "embedding.0.weight":(64,496), "head.weight":(2,64)},
        "SIRE-EEG": {"temporal.0.weight":(8,1,1,15), "temporal.1.weight":(8,1,1,63), "temporal.2.weight":(8,1,1,127), "spatial.0.weight":(16,1,C,1), "spatial.1.weight":(16,1,C,1), "spatial.2.weight":(16,1,C,1), "depth1.weight":(48,1,1,15), "point1.weight":(64,48,1,1), "depth2.weight":(64,1,1,31), "point2.weight":(64,64,1,1), "embedding.0.weight":(64,512), "head.weight":(2,64)},
    }
    output: dict[str, Any] = {}
    for name, model in models.items():
        state = model.state_dict()
        for key, shape in expected[name].items():
            if key not in state or tuple(state[key].shape) != shape: raise RuntimeError(f"state shape audit failed {name}/{key}")
        with torch.inference_mode(): logits, z = model(torch.zeros(1, C, T))
        if tuple(logits.shape) != (1, 2) or tuple(z.shape) != (1, 64): raise RuntimeError(f"forward audit failed: {name}")
        output[name] = {"parameters_C64_T1000_K2":sum(p.numel() for p in model.parameters() if p.requires_grad), "state_shapes":expected[name]}
    if output["SIRE-EEG"]["parameters_C64_T1000_K2"] != 48_074: raise RuntimeError("SIRE C64 parameter assertion failed")
    output["EEGNet"].update({"class":"eegnet_locked.EEGNet(64,1000)", "source":str(EEGNET_SOURCE), "sha256":EXPECTED_SHA256[EEGNET_SOURCE]})
    output["SIRE-EEG"].update({"class":"run_carrier_screen.CompactLite(64,'bn')", "source":str(SIRE_SOURCE), "sha256":EXPECTED_SHA256[SIRE_SOURCE], "formula":"44,872 + 48*C + 65*K = 48,074"})
    return output


def paths(subject: str, session: str, task: str) -> tuple[Path, Path, Path]:
    base = DATA / subject / session / "eeg" / f"{subject}_{session}_task-{task}"
    return base.with_name(base.name + "_eeg.vhdr"), base.with_name(base.name + "_events.tsv"), base.with_name(base.name + "_eeg.json")


def scan_raw() -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, Any]]:
    mne.set_log_level("WARNING")
    downloaded = sorted([item.name for item in DATA.glob("sub-*") if item.is_dir()], key=subject_key)
    if len(downloaded) != 95: raise RuntimeError(f"downloaded subject count {len(downloaded)} != 95")
    cells: list[dict[str, Any]] = []; channel_orders=set(); filter_configs=set(); artifacts=set(); event_codes={task:set() for task,_,_ in TASKS}; counts={subject:{} for subject in downloaded}
    for subject in downloaded:
        for session, session_id, session_label in SESSIONS:
            for task, label, class_name in TASKS:
                vhdr, event_path, metadata_path = paths(subject, session, task)
                if not all(item.is_file() for item in (vhdr,event_path,metadata_path)): raise RuntimeError(f"missing raw M3CV task file {subject}/{session}/{task}")
                with event_path.open(encoding="utf-8", newline="") as handle: events=list(csv.DictReader(handle, delimiter="\t"))
                n=len(events)
                if n <= 0: raise RuntimeError(f"empty event sidecar: {event_path}")
                onset=np.asarray([float(row["onset"]) for row in events]); duration=np.asarray([float(row["duration"]) for row in events]); sample=np.asarray([int(float(row["sample"])) for row in events]); epoch=np.asarray([int(row["epoch_index"]) for row in events])
                if not (np.allclose(onset,np.arange(n)*4.0) and np.allclose(duration,3.0) and np.array_equal(sample,np.arange(n)*T) and np.array_equal(epoch,np.arange(n))): raise RuntimeError(f"task timing mismatch: {event_path}")
                meta=json.loads(metadata_path.read_text(encoding="utf-8")); raw=mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
                if str(meta.get("TaskName")) != task or str(meta.get("RecordingType")) != "continuous" or float(meta.get("SamplingFrequency")) != FS or len(raw.ch_names) != C or float(raw.info["sfreq"]) != FS or raw.n_times != n*T: raise RuntimeError(f"raw schema mismatch: {vhdr}")
                values={str(row["value"]) for row in events}; trials={str(row["trial_type"]) for row in events}; codes={str(row.get("task_code",row["value"])) for row in events}
                if len(values)!=1 or len(trials)!=1 or len(codes)!=1: raise RuntimeError(f"event identifier mismatch: {event_path}")
                channel_orders.add(tuple(raw.ch_names)); filter_configs.add(json.dumps(meta.get("SoftwareFilters",{}),sort_keys=True)); artifacts.add(str(meta.get("SubjectArtefactDescription",""))); event_codes[task].add((next(iter(values)),next(iter(trials)),next(iter(codes))))
                counts[subject][(session,task)] = n
                cells.append({"subject":subject,"session":session,"session_id":session_id,"session_label":session_label,"task":task,"label":label,"class":class_name,"vhdr":vhdr,"events":events,"metadata":meta})
    if len(channel_orders)!=1 or any(len(item)!=1 for item in event_codes.values()) or any("No bad epoch rejection" not in item for item in artifacts): raise RuntimeError("dataset metadata consistency audit failed")
    eligible=[]; excluded=[]
    for subject in downloaded:
        minimum=min(counts[subject].values())
        if minimum >= 8: eligible.append(subject)
        else: excluded.append(subject)
    if len(eligible)!=93 or excluded != ["sub-035","sub-080"]: raise RuntimeError(f"cohort eligibility drift: N={len(eligible)}, excluded={excluded}")
    return downloaded, eligible, cells, {"counts":counts,"channel_names":list(next(iter(channel_orders))),"filters":[json.loads(v) for v in sorted(filter_configs)],"event_codes":{k:list(v)[0] for k,v in event_codes.items()},"artifacts":sorted(artifacts)}


def build_cache(cells: list[dict[str, Any]], eligible: list[str], channel_names: list[str]) -> list[dict[str, Any]]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True); records=[]; grouped={}
    for cell in cells:
        if cell["subject"] in eligible: grouped.setdefault((cell["subject"],cell["session"]),[]).append(cell)
    for (subject,session), pair in sorted(grouped.items(),key=lambda v:(subject_key(v[0][0]),v[0][1])):
        pair.sort(key=lambda item:item["label"])
        if [item["label"] for item in pair] != [0,1]: raise RuntimeError(f"class pairing error: {subject}/{session}")
        target=CACHE_ROOT/subject; target.mkdir(parents=True,exist_ok=True); x_path=target/f"{session}_X.npy"; y_path=target/f"{session}_y.npy"
        expected=sum(len(item["events"]) for item in pair)
        if x_path.is_file() and y_path.is_file():
            x=np.load(x_path,mmap_mode="r",allow_pickle=False); y=np.load(y_path,mmap_mode="r",allow_pickle=False)
            want=np.concatenate([np.full(len(item["events"]),item["label"],dtype=np.int64) for item in pair])
            if x.shape != (expected,C,T) or x.dtype != np.float32 or not np.array_equal(y,want): raise RuntimeError(f"variable cache resume mismatch: {x_path}")
        else:
            temp=x_path.with_suffix(".part.npy"); x=np.lib.format.open_memmap(temp,mode="w+",dtype=np.float32,shape=(expected,C,T)); y=np.empty(expected,dtype=np.int64); cursor=0
            for cell in pair:
                raw=mne.io.read_raw_brainvision(cell["vhdr"],preload=True,verbose="ERROR")
                if tuple(raw.ch_names)!=tuple(channel_names) or float(raw.info["sfreq"])!=FS: raise RuntimeError(f"reader schema drift: {cell['vhdr']}")
                values=raw.get_data().astype(np.float32,copy=False)
                for event_index,event in enumerate(cell["events"]):
                    start=int(float(event["sample"])); value=values[:,start:start+T]
                    if value.shape != (C,T): raise RuntimeError(f"epoch bounds failure: {cell['vhdr']}/{event_index}")
                    x[cursor]=value; y[cursor]=cell["label"]; cursor+=1
            x.flush(); del x
            if cursor != expected: raise RuntimeError("variable cache row count mismatch")
            os.replace(temp,x_path); np.save(y_path,y,allow_pickle=False)
        cursor=0
        for cell in pair:
            for trial,event in enumerate(cell["events"]):
                records.append({"subject":subject,"session":session,"session_id":cell["session_id"],"class":cell["class"],"label":cell["label"],"trial_id":trial,"cache_index":cursor,"shape":"64x1000","dtype":"float32","source_file":str(cell["vhdr"]),"original_event_index":event.get("epoch_index"),"original_epoch_id":event.get("epoch_id"),"cache_file":str(x_path),"label_file":str(y_path)})
                cursor+=1
        write_json(target/f"{session}_cache_metadata.json",{"subject":subject,"session":session,"X":str(x_path),"y":str(y_path),"shape":[expected,C,T],"dtype":"float32","channel_order":channel_names,"class_order":["left_hand","right_hand"],"source_is_native_direct_epoch":True})
    if not records: raise RuntimeError("empty variable cache")
    return records


def load_bundle(stage1: Any, eligible: list[str]) -> Any:
    rows=[]
    for subject in eligible:
        for session,session_id,_ in SESSIONS:
            x_path=CACHE_ROOT/subject/f"{session}_X.npy"; y_path=CACHE_ROOT/subject/f"{session}_y.npy"; x=np.load(x_path,mmap_mode="r",allow_pickle=False); y=np.load(y_path,mmap_mode="r",allow_pickle=False)
            if x.ndim!=3 or x.shape[1:] != (C,T) or x.dtype != np.float32 or y.shape != (x.shape[0],) or set(np.unique(y))!={0,1}: raise RuntimeError(f"cache schema invalid: {x_path}")
            if min(int((y==0).sum()),int((y==1).sum()))<8: raise RuntimeError(f"cache eligibility invalid: {x_path}")
            rows.extend(stage1.core.Row(subject,session_id,str(x_path),i,int(y[i])) for i in range(len(y)))
    accessor=stage1.core.SignalAccessor(rows,CACHE_ROOT,C)
    return stage1.core.DatasetBundle("M3CV",eligible,rows,accessor,C)


def splits(subjects: list[str]) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    perm=[str(v) for v in np.random.default_rng(SEED).permutation(np.asarray(subjects,dtype=object))]; outer_parts=[list(map(str,v.tolist())) for v in np.array_split(np.asarray(perm,dtype=object),5)]; folds=[]; rows=[]
    for fold_id,outer in enumerate(outer_parts):
        remaining=[s for s in subjects if s not in set(outer)]; shuffled=[str(v) for v in np.random.default_rng(1000+fold_id).permutation(np.asarray(remaining,dtype=object))]; val,train=shuffled[:10],shuffled[10:]
        if len(outer) not in (18,19) or len(val)!=10 or len(train)!=(len(subjects)-len(outer)-10) or set(outer)&set(val) or set(outer)&set(train) or set(val)&set(train): raise RuntimeError(f"split invariant failed fold={fold_id}")
        fold={"fold_id":fold_id,"fold_seed":SEED,"inner_split_seed":1000+fold_id,"inner_train_subjects":train,"inner_val_subjects":val,"outer_test_subjects":outer}; folds.append(fold)
        for role,group in (("inner_train",train),("inner_val",val),("outer_test",outer)): rows.extend({"seed":SEED,"fold":fold_id,"role":role,"subject":subject} for subject in group)
    if sorted(sum((f["outer_test_subjects"] for f in folds),[]),key=subject_key)!=subjects: raise RuntimeError("outer coverage failed")
    return folds,rows


def fit_normalizer(bundle: Any, fold: dict[str,Any]) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    indices=bundle.indices(fold["inner_train_subjects"],(1,)); total=np.zeros(C,np.float64); square=np.zeros(C,np.float64); n=0
    for start in range(0,len(indices),64):
        values=bundle.accessor.batch(indices[start:start+64]).astype(np.float64); total+=values.sum((0,2)); square+=np.square(values).sum((0,2)); n+=values.shape[0]*values.shape[2]
    mean=(total/n).astype(np.float32); std=np.sqrt(np.maximum(square/n-np.square(mean.astype(np.float64)),1e-12)).astype(np.float32); digest=hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest(); info={"subjects":sorted(fold["inner_train_subjects"],key=subject_key),"session":"S1=ses-01","session_id":1,"trials":int(len(indices)),"samples_per_channel":int(n),"mean_std_sha256":digest}
    output=ROOT/"normalizers"/f"fold{fold['fold_id']}_s1_normalizer.npz"; output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(output,mean=mean,std=std,metadata=json.dumps(info,sort_keys=True)); info["path"]=str(output)
    return mean,std,info


class OpenBMISemanticProxy:
    def __init__(self,bundle:Any): self.bundle=bundle; self.name="OpenBMI"
    def __getattr__(self,name:str)->Any: return getattr(self.bundle,name)


def manifests(stage1: Any,bundle: Any,fold: dict[str,Any]) -> tuple[list[list[dict[str,Any]]],Path,dict[str,Any]]:
    stage1.RUNTIME=ROOT/"runtime"/"authoritative_manifest_generation"; proxy=OpenBMISemanticProxy(bundle); episode,info=stage1.make_manifest(proxy,{**fold,"outer_dev_subjects":fold["outer_test_subjects"]}); source=Path(info["path"]); payload=json.loads(source.read_text(encoding="utf-8")); payload["dataset"]="M3CV"; payload["sampling"].update({"dataset_adapter":"Direct final run_stage1.make_manifest via OpenBMI semantic proxy; S1=ses-01/session 1 and S2=ses-02/session 2. Only output metadata is relabeled.","support_session_name":"ses-01","query_session_name":"ses-02","source_manifest_sha256_before_metadata_relabel":info["sha256"]}); destination=ROOT/"episode_manifests"/f"fold{fold['fold_id']}.json"; write_json(destination,payload); episode=payload["epochs"]
    expected_steps=max(20,math.ceil(info["legal_source_trials"]/EPISODE_TRIALS))
    if info["steps_per_epoch"]!=expected_steps or len(episode)!=EPOCHS or any(len(items)!=expected_steps for items in episode): raise RuntimeError("authoritative episode count drift")
    train=set(fold["inner_train_subjects"]); val=set(fold["inner_val_subjects"]); outer=set(fold["outer_test_subjects"])
    for epoch in episode:
        for item in epoch:
            support=item["support_subjects"]; query=item["query_subjects"]
            if len(support)!=4 or len(query)!=4 or set(support)&set(query) or not (set(support)|set(query))<=train or (set(support)|set(query))&(val|outer): raise RuntimeError("episode subject audit failed")
            for side,session,subjects in (("support",1,support),("query",2,query)):
                indices=item[f"{side}_indices"]
                if len(indices)!=64: raise RuntimeError("episode total trial audit failed")
                for subject in subjects:
                    selected=[int(v) for v in indices if bundle.search_rows[int(v)].subject==subject]
                    labels=[bundle.search_rows[v].label for v in selected]; sessions=[bundle.search_rows[v].session for v in selected]
                    if len(selected)!=16 or set(sessions)!={session} or labels.count(0)!=8 or labels.count(1)!=8 or len(set(selected))!=16: raise RuntimeError("episode class/no-replacement audit failed")
    return episode,destination,{"fold":fold["fold_id"],"epochs":EPOCHS,"steps_per_epoch":expected_steps,"episodes":EPOCHS*expected_steps,"legal_S1_trials":info["legal_source_trials"],"support_trials":EPOCHS*expected_steps*64,"query_trials":EPOCHS*expected_steps*64,"manifest_sha256":sha256(destination),"source_manifest_sha256":info["sha256"]}


class BatchCache:
    def __init__(self,bundle:Any,mean:np.ndarray,std:np.ndarray,device:torch.device): self.bundle=bundle; self.mean=mean; self.std=std; self.device=device
    def batch(self,indices:Iterable[int])->tuple[torch.Tensor,torch.Tensor]:
        values=np.asarray(list(indices),dtype=np.int64); x=self.bundle.accessor.batch(values).astype(np.float32,copy=False); x=(x-self.mean[None,:,None])/np.maximum(self.std[None,:,None],1e-6); y=self.bundle.labels(values)
        return torch.from_numpy(np.ascontiguousarray(x)).to(self.device,non_blocking=True),torch.as_tensor(y,dtype=torch.long,device=self.device)


def construct(name:str,eegnet:Any,sire:Any)->torch.nn.Module:
    if name=="EEGNet": return eegnet.EEGNet(C,T)
    if name=="SIRE-EEG": return sire.CompactLite(C,"bn")
    raise ValueError(name)


def eval_session(model:torch.nn.Module,bundle:Any,cache:BatchCache,subjects:list[str],session:int)->dict[str,dict[str,float]]:
    model.eval(); result={}
    with torch.no_grad():
        for subject in subjects:
            indices=bundle.indices([subject],(session,)); y=bundle.labels(indices); chunks=[]
            if len(indices)==0: raise RuntimeError(f"no outer trials: {subject}/S{session}")
            for start in range(0,len(indices),128): chunks.append(model(cache.batch(indices[start:start+128])[0])[0].float().cpu().numpy())
            pred=np.concatenate(chunks).argmax(1); result[subject]={"BA":float(balanced_accuracy_score(y,pred)),"Macro_F1":float(f1_score(y,pred,average="macro",zero_division=0)),"accuracy":float(accuracy_score(y,pred)),"trials":int(len(y))}
    return result


def train_one(name:str,fold:dict[str,Any],manifest:list[list[dict[str,Any]]],manifest_path:Path,bundle:Any,cache:BatchCache,eegnet:Any,sire:Any,models:dict[str,Any],norm:dict[str,Any])->dict[str,Any]:
    set_seed(SEED); model=construct(name,eegnet,sire).to(cache.device); params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params!=models[name]["parameters_C64_T1000_K2"] or (name=="SIRE-EEG" and params!=48_074): raise RuntimeError(f"per-fold parameter audit failure: {name}")
    set_seed(TRAINING_SEED); directory=ROOT/"runtime"/"checkpoints"/name/f"fold{fold['fold_id']}_seed0"; directory.mkdir(parents=True,exist_ok=True); latest=directory/"checkpoint_latest.pt"; selected=directory/"selected_best.pt"; init=state_hash(copy.deepcopy(model.state_dict())); manifest_hash=sha256(manifest_path); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY); amp=cache.device.type=="cuda"; scaler=torch.amp.GradScaler("cuda",enabled=amp); start=1; history=[]; best=-float("inf"); best_epoch=None; best_state=None
    if latest.exists():
        saved=torch.load(latest,map_location=cache.device,weights_only=False)
        if saved["init_sha256"]!=init or saved["manifest_sha256"]!=manifest_hash or saved["normalizer_sha256"]!=norm["mean_std_sha256"]: raise RuntimeError(f"unsafe resume: {latest}")
        model.load_state_dict(saved["current_state"]); opt.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"]); restore_rng(saved["rng"]); start=int(saved["epoch"])+1; history=saved["history"]; best=saved["best_val_BA"]; best_epoch=saved["best_epoch"]; best_state=saved["best_state"]
    started=time.perf_counter()
    stale_epochs = 0
    if history and best_epoch is not None:
        stale_epochs = sum(1 for row in reversed(history[int(best_epoch):]) if not row["selected"])
    for epoch in range(start,EPOCHS+1):
        model.train(); losses=[]
        for item in manifest[epoch-1]:
            x,y=cache.batch(item["support_indices"]+item["query_indices"]); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=cache.device.type,dtype=torch.float16,enabled=amp): logits,_=model(x); loss=F.cross_entropy(logits,y)
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite CE: {name}/fold{fold['fold_id']}")
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),GRADIENT_CLIP); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        validation=eval_session(model,bundle,cache,fold["inner_val_subjects"],2); val_ba=float(np.mean([row["BA"] for row in validation.values()])); chosen=epoch>=MIN_EPOCH and val_ba>best+1e-12
        if chosen:
            best,best_epoch,best_state=val_ba,epoch,copy.deepcopy(model.state_dict())
            stale_epochs=0
        elif epoch >= MIN_EPOCH:
            stale_epochs += 1
        early_stop = epoch >= MIN_EPOCH and stale_epochs >= EARLY_STOP_PATIENCE
        history.append({"epoch":epoch,"CE":float(np.mean(losses)),"inner_val_S2_subject_BA":val_ba,"selected":bool(chosen),"stale_epochs":stale_epochs,"early_stop":bool(early_stop)}); torch.save({"epoch":epoch,"history":history,"best_val_BA":best,"best_epoch":best_epoch,"best_state":best_state,"current_state":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"rng":rng_state(),"manifest_sha256":manifest_hash,"init_sha256":init,"normalizer_sha256":norm["mean_std_sha256"]},latest)
        if epoch==1 or epoch%5==0 or chosen: print(f"[{name} fold={fold['fold_id']} seed=0] epoch={epoch:02d} CE={history[-1]['CE']:.4f} valS2BA={val_ba:.4f}",flush=True)
        if early_stop:
            print(f"[{name} fold={fold['fold_id']} seed=0] early-stop epoch={epoch:02d} after {EARLY_STOP_PATIENCE} strict non-improvements",flush=True)
            break
    if best_state is None or best_epoch is None: raise RuntimeError("no eligible checkpoint")
    model.load_state_dict(best_state); torch.save(model.state_dict(),selected); result={"model":name,"fold":fold["fold_id"],"seed":SEED,"selected_epoch":int(best_epoch),"best_inner_val_S2_subject_BA":float(best),"checkpoint_path":str(selected),"checkpoint_sha256":sha256(selected),"latest_checkpoint_path":str(latest),"init_sha256":init,"manifest_sha256":manifest_hash,"normalizer_sha256":norm["mean_std_sha256"],"parameter_count":params,"amp":amp,"max_epochs":EPOCHS,"min_epoch":MIN_EPOCH,"early_stop_patience":EARLY_STOP_PATIENCE,"epochs_completed":len(history),"early_stopped":bool(len(history)<EPOCHS),"elapsed_seconds":time.perf_counter()-started,"history":history,"inner_train_subjects":fold["inner_train_subjects"],"inner_val_subjects":fold["inner_val_subjects"],"outer_test_subjects":fold["outer_test_subjects"]}; del model
    if cache.device.type=="cuda": torch.cuda.empty_cache()
    return result


def subject_summary(rows:list[dict[str,Any]],model:str,subjects:list[str])->tuple[dict[str,float],dict[str,dict[str,float]]]:
    result={}
    for subject in subjects:
        data={row["session"]:row for row in rows if row["model"]==model and row["subject"]==subject}
        if set(data)!={"S1","S2"}: raise RuntimeError(f"subject metric coverage failed: {model}/{subject}")
        result[subject]={"future S2 BA":float(data["S2"]["BA"]),"future S2 Macro-F1":float(data["S2"]["Macro_F1"]),"WS-BA":min(float(data["S1"]["BA"]),float(data["S2"]["BA"]))}
    return {metric:float(np.mean([value[metric] for value in result.values()])) for metric in ("future S2 BA","future S2 Macro-F1","WS-BA")},result


def bootstrap(delta:np.ndarray)->dict[str,Any]:
    draws=delta[np.random.default_rng(SEED).integers(0,len(delta),size=(20_000,len(delta)))].mean(1)
    return {"subjects":int(len(delta)),"resamples":20_000,"mean":float(delta.mean()),"median":float(np.median(delta)),"ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),"improved":int((delta>1e-12).sum()),"tied":int((np.abs(delta)<=1e-12).sum()),"harmed":int((delta<-1e-12).sum())}


def distribution(values:list[int])->str: return f"min={min(values)}, median={float(np.median(values)):.1f}, mean={float(np.mean(values)):.2f}, max={max(values)}"


def write_pretraining_reports(downloaded:list[str],eligible:list[str],raw:dict[str,Any],cache_rows:list[dict[str,Any]],models:dict[str,Any],folds:list[dict[str,Any]],episode_audits:list[dict[str,Any]],normalizers:dict[int,dict[str,Any]])->None:
    counts=raw["counts"]; excluded=[s for s in downloaded if s not in eligible]
    lines=["# M3CV matched episodic protocol","","This is a seed-0 external replication on M3CV / NEMAR nm000166. The task is left-hand versus right-hand motor execution, not motor imagery.","",f"- Downloaded cohort: 95. Analysis cohort: 93 participants with at least eight valid trials per class in both sessions. Excluded for pre-training availability only: {', '.join(excluded)}.","- S1=`ses-01`; S2=`ses-02`; 64 EEG channels; 250 Hz; direct native 4-s segments, T=1000. No resampling, window search, CSP/xDAWN/Riemannian or handcrafted features.","- Cache retains every valid LH/RH source epoch for every eligible subject/session. Variable subject/session counts are preserved; outer evaluation uses all valid cached trials.","- Fixed seed-0 five-fold outer subject CV: 19,19,19,18,18 outer subjects; 10 inner validation; remaining 64/65 inner train. EEGNet and SIRE use identical splits/manifests.","- The direct final sampler supplies 4 S1 support and 4 different S2 query inner-train subjects, 8 LH+8 RH per subject: 64+64=128 ordinary-CE trials. It permits repetitions only across episodes.","- Normalization pools all valid S1 epochs from inner-train subjects, exactly as the authoritative implementation; variable trial counts therefore give subjects proportional trial weighting.","- User-authorized training correction: retain the final AdamW/ordinary-CE/AMP/clip settings and 60-epoch maximum, but train at least 10 epochs then stop after 8 consecutive strict non-improvements in inner-val S2 subject-equal BA. Strict `>` preserves earliest-tie checkpoint selection.","- Hard stop after this corrected seed-0 execution: no other seed, diagnostic, ablation, ScaleCollapse, rank matching, PRD, BN-state, adaptation, suppression, baseline or tuning."]
    write_text(ROOT/"M3CV_PROTOCOL.md",lines)
    audit=["# M3CV dataset audit","", "- Dataset: M3CV / NEMAR `nm000166` v1.0.0, DOI `10.82901/nemar.nm000166`.","- Source manifest: `https://data.nemar.org/nm000166/v1.0.0/manifest.json`; local downloader audit: `/root/m3cv_nm000166_external_replication/M3CV_DOWNLOAD_AUDIT.md`.","- BIDS BrainVision `.vhdr`/`.eeg`/`.vmrk` plus BIDS event and EEG JSON sidecars; `RecordingType=continuous` is documented by the source as pseudo-continuous reconstruction of native 4-s epochs.","- All 380 task recordings passed: C=64, fs=250 Hz, each valid epoch T=1000, event duration=3 s, onset/sample sequence is 4 s/1000 samples within its recording. Trial counts naturally vary.",f"- Task identifiers (value, trial_type, task_code): `motorLHand`={raw['event_codes']['motorLHand']}; `motorRHand`={raw['event_codes']['motorRHand']}.","", "## Channel order","",", ".join(f"`{v}`" for v in raw["channel_names"]),"", "## Downloaded/eligible IDs","",f"- Downloaded 95: {', '.join(f'`{v}`' for v in downloaded)}",f"- Eligible 93: {', '.join(f'`{v}`' for v in eligible)}",f"- Excluded availability-only: {', '.join(f'`{v}`' for v in excluded)}"]
    write_text(ROOT/"DATASET_AUDIT.md",audit)
    cohort=["# Cohort eligibility audit","", "Eligibility was fixed before episode creation, model initialization, training, checkpoint selection or outer evaluation.", "", "Criterion: at least 8 valid trials in every session x class cell. Manuscript-facing cohort: **93 participants with at least eight valid trials per class in both sessions.**", "", "| Subject | S1 LH | S1 RH | S2 LH | S2 RH | Minimum | Eligible | Exclusion reason |", "|---|---:|---:|---:|---:|---:|---|---|"]
    for subject in downloaded:
        a=counts[subject]; value=[a[("ses-01","motorLHand")],a[("ses-01","motorRHand")],a[("ses-02","motorLHand")],a[("ses-02","motorRHand")]]; ok=subject in eligible; reason="" if ok else "pre-training availability: one or more session x class cells has fewer than 8 valid trials"
        cohort.append(f"| {subject} | {value[0]} | {value[1]} | {value[2]} | {value[3]} | {min(value)} | {'yes' if ok else 'no'} | {reason} |")
    write_text(ROOT/"COHORT_ELIGIBILITY_AUDIT.md",cohort)
    by_session={}
    for session,_,_ in SESSIONS:
        for task,_,class_name in TASKS: by_session[f"{session}/{class_name}"]=[counts[s][(session,task)] for s in eligible]
    prep=["# M3CV preprocessing audit","", "No new signal preprocessing was performed. The cache reads every valid native 4-s segment directly from BrainVision: no resampling, filtering, rereferencing, cropping, artifact rejection, channel reordering, duplication, truncation or cache normalization.","",f"- Source SoftwareFilters (recorded, not applied by this runner): `{json.dumps(raw['filters'],sort_keys=True)}`.","- Sidecars document linked TP9/TP10 mastoid reference, AFz ground, ICA visual eye-artifact processing, source-level bad-channel interpolation when applicable, and no bad-epoch rejection.","- MNE BrainVision calibration decodes the file into physical values; it is format decoding, not a newly introduced filter or normalization.","", "## Eligible-cache valid-trial distributions"]
    prep.extend(f"- `{key}`: {distribution(values)} (N subjects={len(values)})." for key,values in by_session.items())
    write_text(ROOT/"PREPROCESSING_AUDIT.md",prep)
    model_lines=["# Authoritative model audit","", "Both models are imported directly after SHA-256 validation. No model class is copied or reconstructed in this experiment.","", "| Model | Final class | SHA-256 | C=64,T=1000,K=2 parameters |", "|---|---|---|---:|"]
    for name in ("EEGNet","SIRE-EEG"): model_lines.append(f"| {name} | `{models[name]['class']}` | `{models[name]['sha256']}` | {models[name]['parameters_C64_T1000_K2']:,} |")
    model_lines.extend(["", "SIRE is the final CompactLite BN architecture: temporal k=15/63/127 branches (1->8), branch spatial grouped 8->16, concatenate 48, depthwise k=15 48->48 then pointwise 48->64, depthwise k=31 64->64 then pointwise 64->64, adaptive 8 bins, 512->64 embedding, ELU, LayerNorm(64), and 64->2 head. The per-fold assertion is `44,872 + 48*64 + 65*2 = 48,074`.","", "## Critical state tensor shapes"])
    for name in ("EEGNet","SIRE-EEG"):
        model_lines.append(f"### {name}"); model_lines.extend(f"- `{k}`: `{tuple(v)}`" for k,v in models[name]["state_shapes"].items())
    model_lines.extend(["",f"- Final episode source: `{EPISODE_SOURCE}` SHA-256 `{EXPECTED_SHA256[EPISODE_SOURCE]}`.",f"- Final core source: `{CORE_SOURCE}` SHA-256 `{EXPECTED_SHA256[CORE_SOURCE]}`.",f"- Final training config: `{TRAINING_CONFIG}` SHA-256 `{EXPECTED_SHA256[TRAINING_CONFIG]}`.",f"- Final training source: `{TRAINING_SOURCE}` SHA-256 `{EXPECTED_SHA256[TRAINING_SOURCE]}`.","- Inspected final repository commit: `cf1db5a6d8f337626544b44f13057e86abe54dc8` on `codex/persist-eeg-final-heldout-confirmation-v1`."])
    write_text(ROOT/"AUTHORITATIVE_MODEL_AUDIT.md",model_lines)
    ep=["# Episode manifest audit","", "The final `run_stage1.make_manifest` is invoked directly with only an M3CV session-semantic adapter: source support session 1 maps to `ses-01`, and future query session 2 maps to `ses-02`. The stored JSON is metadata-relabeled M3CV but its episode indices derive from the authoritative sampler.","", "| Fold | Outer | Inner val | Inner train | Legal S1 trials | Steps/epoch | Episodes | Support/query trials | Final manifest SHA-256 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for fold,item in zip(folds,episode_audits): ep.append(f"| {fold['fold_id']} | {len(fold['outer_test_subjects'])} | {len(fold['inner_val_subjects'])} | {len(fold['inner_train_subjects'])} | {item['legal_S1_trials']} | {item['steps_per_epoch']} | {item['episodes']} | {item['support_trials']}/{item['query_trials']} | `{item['manifest_sha256']}` |")
    ep.extend(["", "Each entry was checked: 4 support and 4 different query subjects; all inner-train only; support S1; query S2; 8 LH and 8 RH distinct trials per subject/session/class draw; 64+64 trials. The same files are used for both models."])
    write_text(ROOT/"EPISODE_MANIFEST_AUDIT.md",ep)
    leak=["# Leakage and fairness audit","", "- Eligibility comes exclusively from pre-training trial availability. Its two exclusions are fixed before models or results exist.","- Splits are frozen before optimizer steps; all outer partitions are subject-disjoint and each eligible biological subject appears once in outer test.","- Episode supports/queries are subject-disjoint and inner-train only. The identical frozen fold manifest feeds EEGNet and SIRE.","- Normalization pools all valid S1 epochs from inner train, matching the authoritative pooling behavior. This intentionally weights subjects in proportion to valid S1 trial count; it never uses S2, validation or outer signals.","- Selection consumes only mean inner-val S2 subject-equal BA. Outer labels are never used for training, normalization, checkpoint selection or model choices.","- Outer metrics are subject-equal aggregates, while each subject's metric uses all of its valid cached trials; the 8-trial rule applies only within episodes.","", "## Fold normalizer provenance"]
    leak.extend(f"- Fold {fold}: `{value['mean_std_sha256']}`; {value['trials']} pooled S1 epochs." for fold,value in sorted(normalizers.items()))
    write_text(ROOT/"LEAKAGE_AND_FAIRNESS_AUDIT.md",leak)


def write_final_report(summary:dict[str,dict[str,float]],contrast:dict[str,dict[str,Any]],records:list[dict[str,Any]],outer_fold:dict[tuple[str,int],float],cache_rows:list[dict[str,Any]])->None:
    lines=["# M3CV seed-0 result summary","", "Downloaded cohort = 95. Eligible analysis cohort = 93 participants with at least eight valid trials per class in both sessions. Excluded by this pre-training availability criterion only: `sub-035`, `sub-080`.", "", f"Variable-length cache: {len(cache_rows):,} valid native [64,1000] epochs across eligible people; outer evaluation consumes all valid cached trials per subject/session.", "", "| Model | Future S2 BA | Future S2 Macro-F1 | WS-BA |", "|---|---:|---:|---:|"]
    for name in ("EEGNet","SIRE-EEG"): lines.append(f"| {name} | {summary[name]['future S2 BA']:.4f} | {summary[name]['future S2 Macro-F1']:.4f} | {summary[name]['WS-BA']:.4f} |")
    lines.extend(["", "## Paired SIRE-EEG minus EEGNet (N=93, 20,000 subject bootstrap draws)", "", "| Metric | Mean delta | 95% CI | Improved / tied / harmed |", "|---|---:|---:|---:|"])
    for metric,item in contrast.items(): lines.append(f"| {metric} | {item['mean']:+.4f} ({item['mean']*100:+.2f} pp) | [{item['ci_low']:+.4f}, {item['ci_high']:+.4f}] | {item['improved']} / {item['tied']} / {item['harmed']} |")
    lines.extend(["", "## Selected checkpoints", "", "| Fold | Model | Selected epoch | Inner-val S2 BA | Outer S2 BA | Checkpoint SHA-256 |", "|---:|---|---:|---:|---:|---|"])
    for item in sorted(records,key=lambda v:(v["fold"],v["model"])): lines.append(f"| {item['fold']} | {item['model']} | {item['selected_epoch']} | {item['best_inner_val_S2_subject_BA']:.4f} | {outer_fold[(item['model'],item['fold'])]:.4f} | `{item['checkpoint_sha256']}` |")
    lines.extend(["", "Training used the user-authorized maximum-60/minimum-10/patience-8 early-stopping correction, retaining strict earliest-tie selection by inner-val S2 subject-equal BA.", "", "Hard stop honored: seed 0 only; no PEEH, PSWA, ablation, ScaleCollapse, rank-matched control, PRD, BN-state, adaptation, suppression, other model, tuning or additional seed was run."])
    write_text(ROOT/"SEED0_RESULT_SUMMARY.md",lines)


def run()->None:
    ROOT.mkdir(parents=True,exist_ok=True); stage1,eegnet,sire,recipe=load_authoritative(); models=audit_models(eegnet,sire); downloaded,eligible,cells,raw=scan_raw(); cache_rows=build_cache(cells,eligible,raw["channel_names"]); write_csv(ROOT/"CACHE_MANIFEST.csv",cache_rows,["subject","session","session_id","class","label","trial_id","cache_index","shape","dtype","source_file","original_event_index","original_epoch_id","cache_file","label_file"]); bundle=load_bundle(stage1,eligible); folds,split_rows=splits(eligible); write_csv(ROOT/"subject_split_manifest.csv",split_rows,["seed","fold","role","subject"])
    normalizers={}; all_manifests={}; manifest_paths={}; episode_audits=[]
    for fold in folds:
        normalizers[fold["fold_id"]]=fit_normalizer(bundle,fold); episode,path,audit=manifests(stage1,bundle,fold); all_manifests[fold["fold_id"]]=episode; manifest_paths[fold["fold_id"]]=path; episode_audits.append(audit)
    write_pretraining_reports(downloaded,eligible,raw,cache_rows,models,folds,episode_audits,{k:v[2] for k,v in normalizers.items()}); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); records=[]
    for fold in folds:
        mean,std,norm=normalizers[fold["fold_id"]]; cache=BatchCache(bundle,mean,std,device)
        for name in ("EEGNet","SIRE-EEG"): records.append(train_one(name,fold,all_manifests[fold["fold_id"]],manifest_paths[fold["fold_id"]],bundle,cache,eegnet,sire,models,norm))
        if device.type=="cuda": torch.cuda.empty_cache()
    write_json(ROOT/"seed0_training_records.json",{"seed":SEED,"device":str(device),"final_repository":str(FINAL_REPO),"final_repository_commit":"cf1db5a6d8f337626544b44f13057e86abe54dc8","training_recipe":recipe,"downloaded_subjects":downloaded,"eligible_subjects":eligible,"excluded_subjects":["sub-035","sub-080"],"normalizers":{str(k):v[2] for k,v in normalizers.items()},"episode_audits":episode_audits,"records":records})
    index={(r["model"],r["fold"]):r for r in records}; subject_rows=[]; outer_fold={}
    for fold in folds:
        mean,std,norm=normalizers[fold["fold_id"]]; cache=BatchCache(bundle,mean,std,device)
        for name in ("EEGNet","SIRE-EEG"):
            record=index[(name,fold["fold_id"])]; model=construct(name,eegnet,sire).to(device); model.load_state_dict(torch.load(record["checkpoint_path"],map_location=device,weights_only=False))
            for _,session_id,label in SESSIONS:
                values=eval_session(model,bundle,cache,fold["outer_test_subjects"],session_id)
                if label=="S2": outer_fold[(name,fold["fold_id"])]=float(np.mean([v["BA"] for v in values.values()]))
                for subject,metric in values.items(): subject_rows.append({"model":name,"fold":fold["fold_id"],"seed":SEED,"checkpoint_sha256":record["checkpoint_sha256"],"selected_epoch":record["selected_epoch"],"normalizer_sha256":norm["mean_std_sha256"],"subject":subject,"session":label,"session_id":session_id,**metric})
            del model
        if device.type=="cuda": torch.cuda.empty_cache()
    write_csv(ROOT/"seed0_subject_metrics.csv",subject_rows,["model","fold","seed","checkpoint_sha256","selected_epoch","normalizer_sha256","subject","session","session_id","BA","Macro_F1","accuracy","trials"]); summary={}; per_subject={}
    for name in ("EEGNet","SIRE-EEG"): summary[name],per_subject[name]=subject_summary(subject_rows,name,eligible)
    contrast={metric:bootstrap(np.asarray([per_subject["SIRE-EEG"][s][metric]-per_subject["EEGNet"][s][metric] for s in eligible])) for metric in ("future S2 BA","WS-BA")}
    fold_rows=[{"seed":SEED,"fold":r["fold"],"model":r["model"],"outer_subjects":len(folds[r["fold"]]["outer_test_subjects"]),"selected_epoch":r["selected_epoch"],"inner_val_S2_BA":r["best_inner_val_S2_subject_BA"],"outer_S2_BA":outer_fold[(r["model"],r["fold"])],"checkpoint_sha256":r["checkpoint_sha256"],"parameter_count":r["parameter_count"]} for r in records]
    write_csv(ROOT/"seed0_fold_metrics.csv",fold_rows,["seed","fold","model","outer_subjects","selected_epoch","inner_val_S2_BA","outer_S2_BA","checkpoint_sha256","parameter_count"])
    summary_rows=[{"row_type":"model","model":name,"metric":"","mean":"","ci95_low":"","ci95_high":"","improved":"","tied":"","harmed":"","future_S2_BA":summary[name]["future S2 BA"],"future_S2_Macro_F1":summary[name]["future S2 Macro-F1"],"WS_BA":summary[name]["WS-BA"],"subjects":len(eligible),"bootstrap_resamples":""} for name in ("EEGNet","SIRE-EEG")]
    summary_rows.extend({"row_type":"paired_contrast","model":"SIRE-EEG minus EEGNet","metric":metric,"mean":item["mean"],"ci95_low":item["ci_low"],"ci95_high":item["ci_high"],"improved":item["improved"],"tied":item["tied"],"harmed":item["harmed"],"future_S2_BA":"","future_S2_Macro_F1":"","WS_BA":"","subjects":item["subjects"],"bootstrap_resamples":item["resamples"]} for metric,item in contrast.items())
    write_csv(ROOT/"seed0_summary.csv",summary_rows,["row_type","model","metric","mean","ci95_low","ci95_high","improved","tied","harmed","future_S2_BA","future_S2_Macro_F1","WS_BA","subjects","bootstrap_resamples"]); write_final_report(summary,contrast,records,outer_fold,cache_rows); print("M3CV_MATCHED_EPISODIC_SEED0_COMPLETE",flush=True)


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--run",action="store_true"); args=parser.parse_args()
    if not args.run: raise RuntimeError("pass --run; no other experiment mode exists")
    run(); return 0


if __name__=="__main__":
    try: raise SystemExit(main())
    except Exception as error:
        print(f"M3CV_PROTOCOL_INVALID: {type(error).__name__}: {error}",flush=True); raise
