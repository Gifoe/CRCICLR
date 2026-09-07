"""R2EEG Stage-1, seed-0-only, SEARCH-only prospective experiment.

No path in this runner enumerates V8_INTERNAL_HOLDOUT or WBCIC true-outer data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import stage1_core as core
from metrics import classification_metrics, geometry
from model_r2eeg import R2EEG, parameter_count
from relational_loss import binary_prd_explicit, direction_from_centroids, prd_loss

SEED, MAX_EPOCHS, MIN_EPOCHS = 0, 60, 10
LR, WEIGHT_DECAY, GRAD_CLIP, LAMBDA_REL = 3e-4, 5e-4, 5.0, 0.25
EPISODE_TRIALS, BOOTSTRAPS, TIE_TOL = 128, 10000, 1e-8
REPO = Path(os.environ.get("R2EEG_REPO", Path(__file__).resolve().parents[3])).resolve()
EXP = REPO / "experiments" / "persist_eeg_r2eeg_stage1_v1"
CODE, PROTOCOL, OUT = EXP / "code", EXP / "protocol", EXP / "outputs"
RUNTIME = Path(os.environ.get("R2EEG_RUNTIME", "/root/rivermind-data/r2eeg_stage1_runtime")).resolve()
OPENBMI_ROOT = Path(os.environ.get("PERSIST_OPENBMI_CACHE", "/root/rivermind-data/persist_eeg_cache/openbmi/openbmi")).resolve()
WBCIC_ROOT = Path(os.environ.get("PERSIST_WBCIC_CACHE", "/root/rivermind-data/persist_eeg_cache/wbcic/wbcic_epochs")).resolve()
SOURCE_SPLIT = REPO / "experiments" / "persist_eeg_transfer_geometry_stage1_v1" / "protocol" / "STAGE1_SEARCH_CV_SPLIT.json"


def clean(v: Any) -> Any:
    if isinstance(v, Path): return str(v)
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)): return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (np.bool_, bool)): return bool(v)
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha_bytes(value: bytes) -> str: return hashlib.sha256(value).hexdigest()
def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state_hash(state: dict[str, torch.Tensor]) -> str:
    b = io.BytesIO(); torch.save(state, b); return sha_bytes(b.getvalue())


def subject_sort(values: list[str], dataset: str) -> list[str]:
    return sorted(map(str, values), key=lambda s: int(s.replace("sub-", "")))


def load_split() -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]], str]:
    if not SOURCE_SPLIT.is_file(): raise FileNotFoundError(SOURCE_SPLIT)
    raw = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    if raw.get("protocol") != "STAGE1_SEARCH_ONLY" or raw.get("seed") != 0:
        raise RuntimeError("source split is not frozen Stage-1 seed 0")
    result, search = {}, {}
    for dataset, expected in (("OpenBMI", 40), ("WBCIC", 31)):
        item = raw["datasets"][dataset]
        search[dataset] = subject_sort(item["search_subjects"], dataset)
        if len(search[dataset]) != expected: raise RuntimeError(f"{dataset} SEARCH size mismatch")
        folds = item["folds"]
        if len(folds) != 3: raise RuntimeError(f"{dataset} needs exactly 3 frozen folds")
        seen: set[str] = set(); checked = []
        for fold in folds:
            fields = {k: [str(x) for x in fold[k]] for k in ("inner_train_subjects", "inner_val_subjects", "outer_dev_subjects")}
            a, b, c = map(set, (fields["inner_train_subjects"], fields["inner_val_subjects"], fields["outer_dev_subjects"]))
            if a & b or a & c or b & c or not (a | b | c) <= set(search[dataset]):
                raise RuntimeError(f"invalid frozen {dataset} fold {fold.get('fold_id')}")
            seen |= c
            checked.append({"fold_id": int(fold["fold_id"]), "fold_seed": int(fold["fold_seed"]),
                            "inner_split_seed": int(fold["inner_split_seed"]), **fields})
        if seen != set(search[dataset]): raise RuntimeError(f"{dataset} outer-development coverage is not exact")
        result[dataset] = checked
    return result, search, sha_file(SOURCE_SPLIT)


def load_bundle(dataset: str, search: list[str]) -> core.DatasetBundle:
    rows: list[core.Row] = []
    if dataset == "OpenBMI":
        for subject in search:
            for session in (1, 2):
                base = OPENBMI_ROOT / f"sub-{int(subject):02d}" / f"ses-{session}" / "mi_1train"
                signal, labels = base.with_name(base.name + "_signals.npy"), base.with_name(base.name + "_codes.npy")
                if not signal.is_file() or not labels.is_file(): raise FileNotFoundError(f"OpenBMI MI cache missing: {base}")
                x, y = np.load(signal, mmap_mode="r", allow_pickle=False), np.load(labels, mmap_mode="r", allow_pickle=False)
                if x.shape != (100, 62, 1000) or x.dtype != np.float32 or y.shape != (100,):
                    raise RuntimeError(f"OpenBMI schema mismatch: {signal} {x.shape}/{x.dtype}, {labels} {y.shape}")
                codes = set(map(int, np.unique(y)))
                if codes != {1, 2}: raise RuntimeError(f"OpenBMI MI codes must be 1/2: {labels}")
                rows.extend(core.Row(subject, session, str(signal), i, int(y[i]) - 1) for i in range(100))
        expected = 40 * 2 * 100
        if len(rows) != expected: raise RuntimeError("OpenBMI row count mismatch")
        return core.DatasetBundle(dataset, search, rows, core.SignalAccessor(rows, OPENBMI_ROOT, 62), 62)
    for subject in search:
        for session in (0, 1, 2):
            x_path, y_path = WBCIC_ROOT / subject / f"ses-{session}_epochs.npy", WBCIC_ROOT / subject / f"ses-{session}_labels.npy"
            if not x_path.is_file() or not y_path.is_file(): raise FileNotFoundError(f"WBCIC SEARCH cache missing: {subject} session {session}")
            x, y = np.load(x_path, mmap_mode="r", allow_pickle=False), np.load(y_path, mmap_mode="r", allow_pickle=False)
            if x.ndim != 3 or x.shape[1:] != (58, 1000) or x.dtype != np.float16 or y.shape != (x.shape[0],):
                raise RuntimeError(f"WBCIC schema mismatch: {x_path}")
            if set(map(int, np.unique(y))) != {0, 1}: raise RuntimeError(f"WBCIC binary labels required: {y_path}")
            rows.extend(core.Row(subject, session, str(x_path), i, int(y[i])) for i in range(x.shape[0]))
    return core.DatasetBundle(dataset, search, rows, core.SignalAccessor(rows, WBCIC_ROOT, 58), 58)


def source_sessions(dataset: str) -> tuple[int, ...]: return (1,) if dataset == "OpenBMI" else (0, 1)


def normalizer(bundle: core.DatasetBundle, subjects: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices = bundle.indices(subjects, source_sessions(bundle.name))
    total, square, n = np.zeros(bundle.channels, np.float64), np.zeros(bundle.channels, np.float64), 0
    for start in range(0, len(indices), 64):
        x = bundle.accessor.batch(indices[start:start + 64]).astype(np.float64)
        total += x.sum(axis=(0, 2)); square += np.square(x).sum(axis=(0, 2)); n += x.shape[0] * x.shape[2]
    mean = (total / n).astype(np.float32); std = np.sqrt(np.maximum(square / n - mean.astype(np.float64) ** 2, 1e-12)).astype(np.float32)
    return mean, std, {"subjects": subject_sort(subjects, bundle.name), "sessions": list(source_sessions(bundle.name)), "trials": int(len(indices)), "samples_per_channel": int(n), "mean_std_sha256": sha_bytes(mean.tobytes() + std.tobytes())}


def make_manifest(bundle: core.DatasetBundle, fold: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    index = core.build_subject_index(bundle); sessions = source_sessions(bundle.name)
    legal = len(bundle.indices(fold["inner_train_subjects"], sessions))
    steps = max(20, math.ceil(legal / EPISODE_TRIALS)); all_epochs = []
    replacements = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        entries = []
        for step in range(steps):
            rng = np.random.default_rng(fold["fold_seed"] + epoch * 1_000_003 + step * 97)
            subjects = [str(x) for x in rng.permutation(np.array(fold["inner_train_subjects"], dtype=object))]
            support, query = subjects[:4], subjects[4:8]
            if len(support) != 4 or len(query) != 4 or set(support) & set(query): raise RuntimeError("invalid subject-disjoint episode")
            si: list[int] = []; qi: list[int] = []
            for s in support: si.extend(core.sample_subject_trials(index, s, sessions, 8 if bundle.name == "OpenBMI" else 4, rng))
            for s in query: qi.extend(core.sample_subject_trials(index, s, (2,), 8, rng))
            if len(si) != 64 or len(qi) != 64: raise RuntimeError("episode must have exactly 64 support and 64 query trials")
            entries.append({"support_subjects": support, "query_subjects": query, "support_indices": si, "query_indices": qi, "replacement": False})
        all_epochs.append(entries)
    payload = {"dataset": bundle.name, "fold_id": fold["fold_id"], "fold_seed": fold["fold_seed"], "steps_per_epoch": steps,
               "sampling": {"support_sessions": list(sessions), "query_sessions": [2], "support_subjects": 4, "query_subjects": 4,
                            "trials_per_episode": 128, "replacement_entries": replacements}, "epochs": all_epochs}
    path = RUNTIME / "episode_manifests" / f"{bundle.name.lower()}_fold{fold['fold_id']}.json"
    write_json(path, payload); return all_epochs, {"path": str(path), "sha256": sha_file(path), "steps_per_epoch": steps, "legal_source_trials": legal}


def prepare(bundle: core.DatasetBundle, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    x = bundle.accessor.batch(indices).astype(np.float32, copy=False)
    x = (x - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
    return torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True)


def evaluate(model: torch.nn.Module, bundle: core.DatasetBundle, subjects: list[str], sessions: tuple[int, ...], mean: np.ndarray, std: np.ndarray, device: torch.device, reps: bool = False) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    model.eval(); result: dict[str, dict[str, float]] = {}; latent: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for subject in subjects:
            idx = bundle.indices([subject], sessions); y = bundle.labels(idx); logits_all=[]; z_all=[]; zr_all=[]; zs_all=[]
            for start in range(0, len(idx), 128):
                x = prepare(bundle, idx[start:start+128], mean, std, device); logits, z = model(x)
                logits_all.append(logits.float().cpu().numpy()); z_all.append(z.float().cpu().numpy())
                if reps:
                    if not isinstance(model, R2EEG): raise RuntimeError("representation audit is R2EEG-only")
                    zr, zs, _ = model.forward_latents(x); zr_all.append(zr.float().cpu().numpy()); zs_all.append(zs.float().cpu().numpy())
            logits_np, z_np = np.concatenate(logits_all), np.concatenate(z_all); pred = logits_np.argmax(1)
            result[subject] = {**classification_metrics(y, pred), "trials": int(len(y))}
            if reps: latent[subject] = {"y": y, "logits": logits_np, "zrel": np.concatenate(zr_all), "zres": np.concatenate(zs_all), "zfull": z_np}
    return result, latent


def _subject_ids(episode: dict[str, Any], side: str, device: torch.device) -> torch.Tensor:
    # Each manifest construction appends a fixed number of trials per subject.
    names = episode[f"{side}_subjects"]; per_subject = 16
    return torch.arange(len(names), device=device).repeat_interleave(per_subject)


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])


def train(model: torch.nn.Module, name: str, bundle: core.DatasetBundle, fold: dict[str, Any], manifest: list[list[dict[str, Any]]],
          mean: np.ndarray, std: np.ndarray, norm_hash: str, manifest_info: dict[str, Any], init_hash: str, device: torch.device) -> dict[str, Any]:
    path = RUNTIME / "checkpoints" / bundle.name.lower() / f"fold{fold['fold_id']}_{name.lower().replace('-', '_')}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    amp = device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, history, best_ba, best_epoch, best_state = 1, [], -float("inf"), None, None
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved["manifest_sha256"] != manifest_info["sha256"] or saved["normalizer_sha256"] != norm_hash or saved["init_sha256"] != init_hash:
            raise RuntimeError(f"resume invariant mismatch for {path}")
        model.load_state_dict(saved["current_state"]); opt.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        start, history, best_ba, best_epoch, best_state = saved["epoch"] + 1, saved["history"], saved["best_ba"], saved["best_epoch"], saved["best_state"]
        restore_rng(saved["rng"])
    started = time.perf_counter()
    for epoch in range(start, MAX_EPOCHS + 1):
        model.train(); ce_values=[]; rel_values=[]
        for episode in manifest[epoch-1]:
            si, qi = np.asarray(episode["support_indices"], np.int64), np.asarray(episode["query_indices"], np.int64)
            indices = np.concatenate((si, qi)); y = torch.from_numpy(bundle.labels(indices)).to(device)
            opt.zero_grad(set_to_none=True)
            x = prepare(bundle, indices, mean, std, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                if name == "R2EEG-Rel":
                    zr, _, zfull = model.forward_latents(x)
                    logits = model.head(model.dropout(zfull))
                else:
                    logits, zfull = model(x)
                ce = F.cross_entropy(logits, y); rel = torch.zeros((), device=device)
                if name == "R2EEG-Rel":
                    rel = prd_loss(zr[:64], y[:64], _subject_ids(episode, "support", device), zr[64:], y[64:], _subject_ids(episode, "query", device))
                loss = ce + LAMBDA_REL * rel
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss for {name}")
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); scaler.step(opt); scaler.update()
            ce_values.append(float(ce.detach().cpu())); rel_values.append(float(rel.detach().cpu()))
        val, _ = evaluate(model, bundle, fold["inner_val_subjects"], (2,), mean, std, device)
        val_ba = float(np.mean([x["BA"] for x in val.values()])); selected = epoch >= MIN_EPOCHS and val_ba > best_ba + 1e-12
        if selected: best_ba, best_epoch, best_state = val_ba, epoch, copy.deepcopy(model.state_dict())
        # anti-collapse record: mean norm of each subject's class-difference in the final deterministic episode.
        with torch.no_grad():
            ep=manifest[epoch-1][-1]; ii=np.asarray(ep["support_indices"]+ep["query_indices"],np.int64); yy=bundle.labels(ii)
            if isinstance(model, R2EEG): zz,_,_=model.forward_latents(prepare(bundle,ii,mean,std,device))
            else: _,zz=model(prepare(bundle,ii,mean,std,device))
            diffs=[]
            for sid in range(8):
                part=zz[sid*16:(sid+1)*16]; lab=yy[sid*16:(sid+1)*16]; diffs.append(float((part[lab==1].mean(0)-part[lab==0].mean(0)).norm().cpu()))
        row={"epoch":epoch,"CE":float(np.mean(ce_values)),"PRD":float(np.mean(rel_values)),"inner_val_subject_BA":val_ba,"selected":selected,"zrel_subject_direction_norm_mean":float(np.mean(diffs))}; history.append(row)
        torch.save({"epoch":epoch,"history":history,"best_ba":best_ba,"best_epoch":best_epoch,"best_state":best_state,"current_state":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"rng":rng_state(),"manifest_sha256":manifest_info["sha256"],"normalizer_sha256":norm_hash,"init_sha256":init_hash},path)
        if epoch == 1 or epoch % 5 == 0 or selected: print(f"[{bundle.name} f{fold['fold_id']} {name}] e={epoch:02d} CE={row['CE']:.4f} PRD={row['PRD']:.4f} valBA={val_ba:.4f}",flush=True)
    if best_state is None: raise RuntimeError("no selectable epoch after epoch 10")
    model.load_state_dict(best_state); final = path.with_name(path.stem + "_best.pt"); torch.save(model.state_dict(), final)
    return {"model":name,"selected_epoch":best_epoch,"best_inner_val_subject_BA":best_ba,"history":history,"checkpoint_path":str(final),"checkpoint_sha256":sha_file(final),"elapsed_seconds_this_invocation":time.perf_counter()-started,"amp":amp,"steps_per_epoch":manifest_info["steps_per_epoch"]}


def bootstrap(a: np.ndarray, b: np.ndarray, seed: int) -> dict[str, Any]:
    d=a-b; rng=np.random.default_rng(seed); draw=rng.choice(d,size=(BOOTSTRAPS,len(d)),replace=True).mean(1)
    return {"n_subjects":int(len(d)),"resamples":BOOTSTRAPS,"mean_delta_BA":float(d.mean()),"median_delta_BA":float(np.median(d)),"ci_low":float(np.quantile(draw,.025)),"ci_high":float(np.quantile(draw,.975)),"improved":int((d>TIE_TOL).sum()),"tied":int((np.abs(d)<=TIE_TOL).sum()),"harmed":int((d<-TIE_TOL).sum())}


def protocol_tests() -> dict[str, Any]:
    rows=[]
    for c in (62,58):
        m=R2EEG(c); x=torch.randn(4,c,1000); logits,z=m(x); zr,zs,zf=m.forward_latents(x); assert logits.shape==(4,2) and zr.shape==zs.shape==(4,64) and zf.shape==(4,128) and z.shape==(4,128); assert parameter_count(m)<150000; rows.append({"channels":c,"params":parameter_count(m),"shape_test":True})
    torch.manual_seed(3); z=torch.randn(32,64); y=torch.tensor([0]*8+[1]*8+[0]*8+[1]*8); s=torch.tensor([0]*16); q=torch.tensor([1]*16)
    generic=prd_loss(z[:16],y[:16],s,z[16:],y[16:],q); explicit=binary_prd_explicit(z[:16],y[:16],s,z[16:],y[16:],q); assert torch.allclose(generic,explicit,atol=1e-7)
    a,b=torch.randn(64),torch.randn(64); shift=torch.randn(64); assert torch.allclose(direction_from_centroids(a,b),direction_from_centroids(a+shift,b+shift),atol=1e-6)
    assert "centroid_alignment" not in inspect.getsource(prd_loss) and "global" not in inspect.getsource(prd_loss)
    return {"shape_and_parameter":rows,"prd_k2_matches_explicit_binary":True,"translation_invariance":True,"no_global_centroid_alignment_loss":True}


def terminal(aggregates: list[dict[str, Any]], folds: pd.DataFrame) -> str:
    arch_fail=any(x["architecture_gain_pp"] < -0.5 for x in aggregates) or any((folds[folds.dataset==d].architecture_gain_pp<0).all() for d in folds.dataset.unique())
    if arch_fail:return "R2EEG_STAGE1_ARCHITECTURE_FAIL_STOP"
    rel_fail=any(x["relational_gain_pp"]<=0 or x["relational_positive_folds"]==0 for x in aggregates)
    if rel_fail:return "R2EEG_STAGE1_RELATIONAL_FAIL_STOP"
    strong=all(x["relational_gain_pp"]>=1 and x["relational_positive_folds"]>=2 and x["relational_median_subject_delta_pp"]>=0 for x in aggregates)
    if strong:return "R2EEG_STAGE1_STRONG_STOP"
    promising=all(x["relational_gain_pp"]>=.5 and x["relational_positive_folds"]>=2 for x in aggregates)
    return "R2EEG_STAGE1_PROMISING_STOP" if promising else "R2EEG_STAGE1_RELATIONAL_WEAK_STOP"


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--device",default="auto"); ap.add_argument("--validate-only",action="store_true"); args=ap.parse_args()
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else args.device); set_seed(SEED)
    for p in (CODE,PROTOCOL,OUT,RUNTIME): p.mkdir(parents=True,exist_ok=True)
    core.RUNTIME=RUNTIME
    folds,search,split_hash=load_split(); bundles={d:load_bundle(d,search[d]) for d in ("OpenBMI","WBCIC")}
    tests=protocol_tests(); write_json(PROTOCOL/"TESTS.json",tests)
    params={f"R2EEG_{c}ch":parameter_count(R2EEG(c)) for c in (62,58)}
    if any(v>=150000 for v in params.values()): raise RuntimeError("R2EEG parameter cap violated")
    write_json(PROTOCOL/"PARAMETER_COUNTS.json",{"trainable_parameters":params,"under_150k":True})
    cache={"OpenBMI":{"root":OPENBMI_ROOT,"required_task":"MI/train","subjects":search["OpenBMI"],"schema":"float32 [100,62,1000], codes 1/2"},"WBCIC":{"root":WBCIC_ROOT,"subjects":search["WBCIC"],"schema":"float16 [trials,58,1000], labels 0/1"}}
    write_json(PROTOCOL/"CACHE_PROVENANCE.json",cache)
    write_json(PROTOCOL/"STAGE1_SEARCH_CV_SPLIT.json",{"source":str(SOURCE_SPLIT),"source_sha256":split_hash,"seed":0,"datasets":{d:{"search_subjects":search[d],"folds":folds[d]} for d in folds}})
    write_json(PROTOCOL/"HOLDOUT_ISOLATION_AUDIT.json",{"V8_INTERNAL_HOLDOUT_loaded":False,"V8_INTERNAL_HOLDOUT_labels_loaded":False,"WBCIC_outer_loaded":False,"WBCIC_outer_labels_loaded":False,"path_traversal":False,"only_V8_SEARCH_loaded":True,"allowed_subjects":search})
    write_json(PROTOCOL/"STAGE1_CONFIG.json",{"seed":0,"folds":3,"max_epochs":60,"selection_epoch_range":[10,60],"optimizer":{"name":"AdamW","lr":LR,"weight_decay":WEIGHT_DECAY,"grad_clip":GRAD_CLIP},"relation":{"loss":"CE + 0.25*PRD(z_rel)","lambda_rel":LAMBDA_REL,"absolute_centroid_alignment":False},"amp":device.type=="cuda","device":str(device)})
    if args.validate_only:
        checks = {}
        for dataset, bundle in bundles.items():
            fold_checks = []
            for fold in folds[dataset]:
                _, manifest_info = make_manifest(bundle, fold)
                outer = set(map(int, bundle.indices(fold["outer_dev_subjects"])))
                payload = json.loads(Path(manifest_info["path"]).read_text(encoding="utf-8"))
                no_outer = not any(outer & set(e["support_indices"] + e["query_indices"]) for epoch in payload["epochs"] for e in epoch)
                if not no_outer: raise RuntimeError("outer-development sample entered an episode")
                fold_checks.append({"fold": fold["fold_id"], "steps_per_epoch": manifest_info["steps_per_epoch"], "episode_sha256": manifest_info["sha256"], "outer_samples_in_episodes": False})
            checks[dataset] = {"subjects":len(search[dataset]),"rows":len(bundle.search_rows),"folds":fold_checks}
        write_json(PROTOCOL / "VALIDATION.json", {"pass": True, "cache_schema": True, "tests": tests, "episode_isolation": checks})
        print("R2EEG_STAGE1_PROTOCOL_VALID", json.dumps(checks),flush=True); return 0
    fold_rows=[]; subjects=[]; training=[]; geo=[]
    for dataset in ("OpenBMI","WBCIC"):
        bundle=bundles[dataset]
        for fold in folds[dataset]:
            mean,std,norm=normalizer(bundle,fold["inner_train_subjects"]); manifest,minfo=make_manifest(bundle,fold); norm_hash=norm["mean_std_sha256"]
            # Explicitly prove episode indices cannot include the fold's outer-development rows.
            outer=set(map(int,bundle.indices(fold["outer_dev_subjects"]))); assert not any(outer & set(e["support_indices"]+e["query_indices"]) for epoch in manifest for e in epoch)
            set_seed(fold["fold_seed"]); eeg=core.EEGNet(bundle.channels).to(device); eeg_init=state_hash(copy.deepcopy(eeg.state_dict())); set_seed(fold["fold_seed"]+20000); eeg_info=train(eeg,"EEGNet-ERM",bundle,fold,manifest,mean,std,norm_hash,minfo,eeg_init,device); eeg_sub,_=evaluate(eeg,bundle,fold["outer_dev_subjects"],(2,),mean,std,device)
            set_seed(fold["fold_seed"]); template=R2EEG(bundle.channels).to(device); initial=copy.deepcopy(template.state_dict()); r2hash=state_hash(initial)
            erm=R2EEG(bundle.channels).to(device); erm.load_state_dict(initial); set_seed(fold["fold_seed"]+20000); erm_info=train(erm,"R2EEG-ERM",bundle,fold,manifest,mean,std,norm_hash,minfo,r2hash,device); erm_sub,_=evaluate(erm,bundle,fold["outer_dev_subjects"],(2,),mean,std,device,reps=True)
            rel=R2EEG(bundle.channels).to(device); rel.load_state_dict(initial); set_seed(fold["fold_seed"]+20000); rel_info=train(rel,"R2EEG-Rel",bundle,fold,manifest,mean,std,norm_hash,minfo,r2hash,device); rel_sub,rel_future=evaluate(rel,bundle,fold["outer_dev_subjects"],(2,),mean,std,device,reps=True)
            _,erm_future=evaluate(erm,bundle,fold["outer_dev_subjects"],(2,),mean,std,device,reps=True)
            for s in fold["outer_dev_subjects"]:
                _,erm_source=evaluate(erm,bundle,[s],source_sessions(dataset),mean,std,device,reps=True); _,rel_source=evaluate(rel,bundle,[s],source_sessions(dataset),mean,std,device,reps=True)
                for n,src,fut in (("R2EEG-ERM",erm_source[s],erm_future[s]),("R2EEG-Rel",rel_source[s],rel_future[s])):
                    for latent_name in ("zrel","zres","zfull"): geo.append({"dataset":dataset,"fold":fold["fold_id"],"subject_id":s,"model":n,"latent":latent_name,**geometry(src[latent_name],src["y"],fut[latent_name],fut["y"],fut["logits"])})
                subjects.append({"dataset":dataset,"fold":fold["fold_id"],"subject_id":s,**{f"eegnet_{k}":v for k,v in eeg_sub[s].items()},**{f"r2erm_{k}":v for k,v in erm_sub[s].items()},**{f"r2rel_{k}":v for k,v in rel_sub[s].items()},"rel_vs_erm_delta_BA_pp":(rel_sub[s]["BA"]-erm_sub[s]["BA"])*100,"erm_vs_eegnet_delta_BA_pp":(erm_sub[s]["BA"]-eeg_sub[s]["BA"])*100})
            mean_metric=lambda d,k:float(np.mean([v[k] for v in d.values()]))
            fold_rows.append({"dataset":dataset,"fold":fold["fold_id"],"EEGNet_BA":mean_metric(eeg_sub,"BA"),"R2EEG_ERM_BA":mean_metric(erm_sub,"BA"),"R2EEG_Rel_BA":mean_metric(rel_sub,"BA"),"EEGNet_macro_F1":mean_metric(eeg_sub,"macro_F1"),"R2EEG_ERM_macro_F1":mean_metric(erm_sub,"macro_F1"),"R2EEG_Rel_macro_F1":mean_metric(rel_sub,"macro_F1"),"architecture_gain_pp":(mean_metric(erm_sub,"BA")-mean_metric(eeg_sub,"BA"))*100,"relational_gain_pp":(mean_metric(rel_sub,"BA")-mean_metric(erm_sub,"BA"))*100,"total_gain_pp":(mean_metric(rel_sub,"BA")-mean_metric(eeg_sub,"BA"))*100,"selected_epoch_EEGNet":eeg_info["selected_epoch"],"selected_epoch_R2EEG_ERM":erm_info["selected_epoch"],"selected_epoch_R2EEG_Rel":rel_info["selected_epoch"],"steps_per_epoch":minfo["steps_per_epoch"],"normalizer_sha256":norm_hash,"episode_manifest_sha256":minfo["sha256"],"r2_initialization_sha256":r2hash})
            training.append({"dataset":dataset,"fold":fold["fold_id"],"normalizer":norm,"episode_manifest":minfo,"init_matching":{"R2EEG_ERM_R2EEG_Rel_same":True,"sha256":r2hash},"models":{"EEGNet-ERM":eeg_info,"R2EEG-ERM":erm_info,"R2EEG-Rel":rel_info}})
            print(f"[{dataset} fold={fold['fold_id']}] EEGNet={fold_rows[-1]['EEGNet_BA']:.4f} R2ERM={fold_rows[-1]['R2EEG_ERM_BA']:.4f} R2Rel={fold_rows[-1]['R2EEG_Rel_BA']:.4f}",flush=True)
    fdf,sdf=pd.DataFrame(fold_rows),pd.DataFrame(subjects); aggregates=[]; boots=[]
    for dataset in ("OpenBMI","WBCIC"):
        sf=sdf[sdf.dataset==dataset]; ff=fdf[fdf.dataset==dataset]; b_rel=bootstrap(sf.r2rel_BA.to_numpy(),sf.r2erm_BA.to_numpy(),SEED); b_arch=bootstrap(sf.r2erm_BA.to_numpy(),sf.eegnet_BA.to_numpy(),SEED+1)
        row={"dataset":dataset,"EEGNet_BA":float(sf.eegnet_BA.mean()),"R2EEG_ERM_BA":float(sf.r2erm_BA.mean()),"R2EEG_Rel_BA":float(sf.r2rel_BA.mean()),"architecture_gain_pp":float((sf.r2erm_BA.mean()-sf.eegnet_BA.mean())*100),"relational_gain_pp":float((sf.r2rel_BA.mean()-sf.r2erm_BA.mean())*100),"total_gain_pp":float((sf.r2rel_BA.mean()-sf.eegnet_BA.mean())*100),"relational_positive_folds":int((ff.relational_gain_pp>0).sum()),"relational_median_subject_delta_pp":b_rel["median_delta_BA"]*100,"relational_ci_low_pp":b_rel["ci_low"]*100,"relational_ci_high_pp":b_rel["ci_high"]*100}; aggregates.append(row); boots.extend([{"comparison":"R2EEG-Rel_minus_R2EEG-ERM",**{"dataset":dataset},**b_rel},{"comparison":"R2EEG-ERM_minus_EEGNet",**{"dataset":dataset},**b_arch}])
    term=terminal(aggregates,fdf); pd.DataFrame(fold_rows).to_csv(OUT/"FOLD_RESULTS.csv",index=False); sdf.to_csv(OUT/"SUBJECT_RESULTS.csv",index=False); pd.DataFrame(aggregates).to_csv(OUT/"AGGREGATE_RESULTS.csv",index=False); pd.DataFrame(geo).to_csv(OUT/"RELATION_GEOMETRY.csv",index=False); write_json(OUT/"BOOTSTRAP.json",boots); write_json(OUT/"TRAINING_LOG.json",training)
    write_json(PROTOCOL/"NORMALIZER_AUDIT.json",[{"dataset":x["dataset"],"fold":x["fold"],**x["normalizer"]} for x in training]); write_json(PROTOCOL/"EPISODE_MANIFEST_HASHES.json",[{"dataset":x["dataset"],"fold":x["fold"],**x["episode_manifest"]} for x in training]); write_json(PROTOCOL/"INITIALIZATION_MATCHING.json",[{"dataset":x["dataset"],"fold":x["fold"],**x["init_matching"]} for x in training]); write_json(PROTOCOL/"INFORMATION_MATCHING.json",{"same_architecture_R2ERM_R2Rel":True,"same_parameterization":True,"same_initialization":True,"same_fold_split":True,"same_normalizer":True,"same_episode_manifest":True,"same_CE_samples_and_order":True,"same_optimizer":True,"same_epochs":True,"only_intended_difference":"PRD present for R2EEG-Rel"})
    decision={"terminal":term,"aggregate":aggregates,"architecture_gate":"pass" if term!="R2EEG_STAGE1_ARCHITECTURE_FAIL_STOP" else "fail","relation_gate":"pass" if term in ("R2EEG_STAGE1_PROMISING_STOP","R2EEG_STAGE1_STRONG_STOP") else "fail_or_weak","holdout_access":False,"next_action":"Do not launch seeds 1/2 automatically."}; write_json(OUT/"DECISION.json",decision); (OUT/"DECISION.md").write_text("# R2EEG Stage-1 decision\n\n`"+term+"`\n\nThis is seed-0 SEARCH-only evidence; no holdout or true outer data were accessed.\n",encoding="utf-8")
    print(term,flush=True); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"R2EEG_STAGE1_PROTOCOL_INVALID: {type(exc).__name__}: {exc}",flush=True); raise
