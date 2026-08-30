"""One-shot locked WBCIC S3 discovery for the six matched controls and ME-HardSCST."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

import v2_common as c
from candidate_engine import margins, match_structured_random, uniform_margin_loss, upper_tail_loss
from mixed_effects import MixedEffectsBank
from source_search import Cache, Geometry, _batch_features, _support_radius, all_features, build_geometry, from_preblock, preblock
from training_components import BankRefreshTracker, EMATeacher, configure_scope, primary_total_loss


METHODS = (
    "ERM", "Mixup", "V1-RandomTransport", "Dynamic-ClassConditional-Uniform-NoKL",
    "Factorized-Uniform-NoKL", "Factorized-HardRandom", "ME-HardSCST",
)


def _load_v1():
    if str(c.STAGE1_CODE) not in sys.path:
        sys.path.insert(0, str(c.STAGE1_CODE))
    spec = importlib.util.spec_from_file_location("me_hard_scst_v1_train_utility", c.STAGE1_CODE / "train_utility.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


V1_TRAIN = _load_v1()


def cache_path(fold: int, seed: int, role: str) -> Path:
    return c.RUNTIME / "discovery_cache" / f"fold-{fold}" / f"seed-{seed}" / f"{role}.npz"


def load_cache(path: Path) -> tuple[Cache, np.ndarray]:
    with np.load(path) as values:
        cache = Cache(values["indices"], values["labels"], values["subjects"].astype(str), values["final"], values["preblock"])
        return cache, values["sessions"]


def build_cache(fold: int, seed: int, device: torch.device, *, lock_verified: bool) -> tuple[Cache, np.ndarray, Cache, np.ndarray]:
    paths = {role: cache_path(fold, seed, role) for role in ("train", "outcome")}
    if all(path.is_file() for path in paths.values()):
        train, train_sessions = load_cache(paths["train"]); outcome, outcome_sessions = load_cache(paths["outcome"])
        return train, train_sessions, outcome, outcome_sessions
    raw, metadata, _ = c.load_development_data("WBCIC")
    role = c.roles("WBCIC", fold)
    source_subjects = tuple(sorted(set(role["model_fit"]) | set(role["validation"])))
    train_idx = c.S1.row_indices(metadata, source_subjects, (0, 1))
    outcome_idx = c.discovery_indices(fold, lock_verified=lock_verified)
    if np.intersect1d(train_idx, outcome_idx).size:
        raise RuntimeError("DISCOVERY_TRAIN_OUTCOME_OVERLAP")
    net, _ = c.load_anchor("ATCNet-CleanRoom", "WBCIC", fold, seed, device)
    net.eval()
    for name, indices in (("train", train_idx), ("outcome", outcome_idx)):
        finals, inputs = [], []
        with torch.inference_mode():
            for start in range(0, len(indices), c.BATCH_SIZE):
                idx = indices[start : start + c.BATCH_SIZE]
                x = torch.from_numpy(c.normalize_raw(raw[idx])).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    before = preblock(net, x); final = from_preblock(net, before)
                inputs.append(before.float().cpu().numpy()); finals.append(final.float().cpu().numpy())
        picked = metadata.iloc[indices]
        path = paths[name]; path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.part")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, indices=indices, labels=picked.label.to_numpy(np.int64), subjects=picked.subject_id.astype(str).to_numpy().astype("U"), sessions=picked.session_id.to_numpy(np.int64), final=np.concatenate(finals).astype(np.float32), preblock=np.concatenate(inputs).astype(np.float32))
        os.replace(temporary, path)
    del net; torch.cuda.empty_cache()
    train, train_sessions = load_cache(paths["train"]); outcome, outcome_sessions = load_cache(paths["outcome"])
    return train, train_sessions, outcome, outcome_sessions


def regate(features: np.ndarray, labels: np.ndarray, subjects: np.ndarray, indices: np.ndarray, offsets: np.ndarray, target: np.ndarray, bank: MixedEffectsBank, device: torch.device) -> Geometry:
    n, budget, dim = offsets.shape
    whitened = np.asarray([[bank.whitened_norm(value) for value in row] for row in offsets], np.float32)
    norm_pass = whitened <= bank.norm_radius + 1e-7
    feature_tensor = torch.from_numpy(features).to(device)
    radius = _support_radius(feature_tensor, labels)
    support_pass = np.zeros((n, budget), bool); semantic_pass = np.zeros((n, budget), bool)
    for start in range(0, n, 32):
        stop = min(n, start + 32)
        candidates = feature_tensor[start:stop, None] + torch.from_numpy(offsets[start:stop]).to(device)
        distance = torch.cdist(candidates.reshape(-1, dim).float(), feature_tensor.float())
        values, near = distance.topk(3, largest=False)
        owner_labels = torch.as_tensor(np.repeat(labels[start:stop], budget), device=device)
        near_labels = torch.as_tensor(labels, device=device)[near]
        support_pass[start:stop] = (values.mean(1) <= torch.as_tensor([radius[int(value)] for value in owner_labels.cpu().tolist()], device=device)).reshape(stop-start, budget).cpu().numpy()
        semantic_pass[start:stop] = (near_labels.eq(owner_labels[:, None]).sum(1) >= 2).reshape(stop-start, budget).cpu().numpy()
    base = support_pass & semantic_pass & norm_pass
    alpha = np.tile(np.asarray(c.ALPHAS, np.float32), c.K_TARGETS)
    return Geometry(offsets, base, support_pass, semantic_pass, norm_pass, whitened, alpha, target, bank)


def random_geometry(structured: Geometry, features: np.ndarray, labels: np.ndarray, subjects: np.ndarray, indices: np.ndarray, fold: int, seed: int, device: torch.device) -> Geometry:
    random_offsets = np.empty_like(structured.offsets)
    for position in range(len(features)):
        for candidate in range(structured.offsets.shape[1]):
            alpha = float(structured.alpha[candidate])
            delta = structured.offsets[position, candidate] / alpha
            rng = np.random.default_rng(c.stable_seed("factorized-hard-random", fold, seed, int(indices[position]), candidate))
            random_offsets[position, candidate] = alpha * structured.bank.hard_random(delta, rng)
    return regate(features, labels, subjects, indices, random_offsets, structured.target, structured.bank, device)


def v1_random_delta(train: Cache, sessions: np.ndarray, fold: int, seed: int) -> np.ndarray:
    rep = {"indices": train.indices, "features": train.final, "labels": train.labels, "subjects": train.subjects, "sessions": sessions, "logits": np.zeros((len(train.labels), 2), np.float32)}
    _, random, _ = V1_TRAIN.frozen_transport(rep, "ATCNet-CleanRoom", fold, seed)
    return random


def unit_path(method: str, fold: int, seed: int) -> Path:
    return c.RUNTIME / "discovery_units" / method / f"fold-{fold}" / f"seed-{seed}"


@torch.no_grad()
def evaluate(net, scope: str, cache: Cache, fold: int, seed: int, method: str, device: torch.device) -> pd.DataFrame:
    net.eval(); logits = []
    for start in range(0, len(cache.labels), c.BATCH_SIZE * 2):
        positions = np.arange(start, min(len(cache.labels), start + c.BATCH_SIZE * 2))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = _batch_features(net, scope, cache, positions, device); logits.append(net.head(h).float().cpu().numpy())
    logits = np.concatenate(logits); pred = logits.argmax(1)
    stable = logits.astype(np.float64) - logits.max(1, keepdims=True); probability = np.exp(stable); probability /= probability.sum(1, keepdims=True)
    rows = []
    for subject in c.subject_sort(np.unique(cache.subjects)):
        mask = cache.subjects.astype(str) == subject
        rows.append({"model": "ATCNet-CleanRoom", "method": method, "fold": fold, "seed": seed, "subject_id": subject, "BA": float(balanced_accuracy_score(cache.labels[mask], pred[mask])), "macro_F1": float(f1_score(cache.labels[mask], pred[mask], average="macro", zero_division=0)), "CE": float(log_loss(cache.labels[mask], probability[mask], labels=[0, 1])), "trials": int(mask.sum()), "future_session": 2})
    return pd.DataFrame(rows)


def train_method(method: str, fold: int, seed: int, train: Cache, sessions: np.ndarray, outcome: Cache, scope: str, q: float, lam: float, device: torch.device) -> pd.DataFrame:
    directory = unit_path(method, fold, seed); result_path = directory / "per_subject.csv"
    if result_path.is_file(): return pd.read_csv(result_path)
    c.set_seed(c.stable_seed("me-hard-discovery", method, fold, seed))
    net, _ = c.load_anchor("ATCNet-CleanRoom", "WBCIC", fold, seed, device)
    parameters = configure_scope("ATCNet-CleanRoom", net, scope)
    optimizer = torch.optim.AdamW(parameters, lr=c.LEARNING_RATE, weight_decay=c.WEIGHT_DECAY)
    dynamic = method in ("Dynamic-ClassConditional-Uniform-NoKL", "Factorized-Uniform-NoKL", "Factorized-HardRandom", "ME-HardSCST")
    teacher = EMATeacher(net, c.EMA_DECAY) if dynamic else None
    refresh = BankRefreshTracker(); previous_b = None; bank_stability = []
    v1_noise = v1_random_delta(train, sessions, fold, seed) if method == "V1-RandomTransport" else None
    order_rng = np.random.default_rng(c.stable_seed("me-hard-discovery-order", fold, seed))
    method_rng = np.random.default_rng(c.stable_seed("me-hard-discovery-method", method, fold, seed))
    geometry = random = None; final_audit = []
    for epoch in range(c.EPOCHS):
        net.train()
        if dynamic:
            if scope == "A" and geometry is not None:
                teacher_features = train.final
            else:
                refresh.refresh(epoch); teacher_features = all_features(teacher.model, scope, train, device)
                factorized = method != "Dynamic-ClassConditional-Uniform-NoKL"
                geometry = build_geometry(teacher_features, train.labels, train.subjects, train.indices, "WBCIC-discovery", fold, seed, device, factorized=factorized)
                if method in ("Factorized-HardRandom", "ME-HardSCST"):
                    random = random_geometry(geometry, teacher_features, train.labels, train.subjects, train.indices, fold, seed, device)
                if previous_b is not None:
                    num = np.sum(previous_b * geometry.bank.full.b, axis=1); den = np.linalg.norm(previous_b, axis=1) * np.linalg.norm(geometry.bank.full.b, axis=1)
                    bank_stability.append(float(np.mean(num / np.maximum(den, 1e-8))))
                previous_b = geometry.bank.full.b.copy()
            if scope == "A" and refresh.refreshes == 0:
                refresh.refresh(0); teacher_features = train.final
                factorized = method != "Dynamic-ClassConditional-Uniform-NoKL"
                geometry = build_geometry(teacher_features, train.labels, train.subjects, train.indices, "WBCIC-discovery", fold, seed, device, factorized=factorized)
                if method in ("Factorized-HardRandom", "ME-HardSCST"):
                    random = random_geometry(geometry, teacher_features, train.labels, train.subjects, train.indices, fold, seed, device)
        epoch_order = order_rng.permutation(len(train.labels)); final_audit = []
        for start in range(0, len(train.labels), c.BATCH_SIZE):
            positions = epoch_order[start:start+c.BATCH_SIZE]
            y = torch.as_tensor(train.labels[positions], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = _batch_features(net, scope, train, positions, device); clean_logits = net.head(h); clean_loss = F.cross_entropy(clean_logits.float(), y); loss = clean_loss
                if method == "Mixup":
                    perm = torch.as_tensor(method_rng.permutation(len(positions)), device=device); weight = float(method_rng.beta(.4, .4))
                    mixed = weight*h + (1-weight)*h[perm]; target = weight*F.one_hot(y,2).float() + (1-weight)*F.one_hot(y[perm],2).float()
                    loss = clean_loss + .5 * (-(target * torch.log_softmax(net.head(mixed).float(), -1)).sum(-1).mean())
                elif method == "V1-RandomTransport":
                    moved = h + torch.from_numpy(v1_noise[positions]).to(device); loss = clean_loss + .5*F.cross_entropy(net.head(moved).float(), y)
                elif dynamic:
                    chosen = random if method == "Factorized-HardRandom" else geometry
                    offsets = torch.from_numpy(chosen.offsets[positions]).to(device).detach(); teacher_clean = torch.from_numpy(teacher_features[positions]).to(device)
                    teacher_logits = teacher.model.head((teacher_clean[:,None]+offsets).flatten(0,1)); owner = torch.arange(len(positions),device=device).repeat_interleave(offsets.shape[1]); owner_y = y[owner]
                    valid = torch.from_numpy(chosen.base_valid[positions]).to(device).reshape(-1) & teacher_logits.detach().argmax(1).eq(owner_y) & margins(teacher_logits.detach(), owner_y).gt(0)
                    if method in ("Factorized-HardRandom", "ME-HardSCST"):
                        other = geometry if method == "Factorized-HardRandom" else random
                        other_offsets = torch.from_numpy(other.offsets[positions]).to(device).detach(); other_logits = teacher.model.head((teacher_clean[:,None]+other_offsets).flatten(0,1))
                        other_valid = torch.from_numpy(other.base_valid[positions]).to(device).reshape(-1) & other_logits.detach().argmax(1).eq(owner_y) & margins(other_logits.detach(), owner_y).gt(0)
                        matched = torch.zeros_like(valid)
                        for local in range(len(positions)):
                            here = owner.eq(local); left = torch.nonzero(valid & here).flatten(); right = torch.nonzero(other_valid & here).flatten(); count=min(len(left),len(right))
                            if count:
                                rng=np.random.default_rng(c.stable_seed("match",method,fold,seed,epoch,int(train.indices[positions[local]])))
                                matched[torch.as_tensor(rng.choice(left.cpu().numpy(),count,replace=False),device=device)]=True
                        valid = valid & matched
                    candidate_logits = net.head((h[:,None]+offsets).flatten(0,1))
                    if method in ("Dynamic-ClassConditional-Uniform-NoKL", "Factorized-Uniform-NoKL"):
                        cf, audit = uniform_margin_loss(clean_logits,candidate_logits,y,owner,valid)
                    else:
                        cf, audit = upper_tail_loss(clean_logits,candidate_logits,y,owner,valid,q=q)
                    loss = primary_total_loss(clean_logits,y,cf,lam)
                    if epoch == c.EPOCHS-1:
                        correct=clean_logits.detach().argmax(1).eq(y)
                        for local,global_pos in enumerate(positions):
                            mask=valid&owner.eq(local)
                            final_audit.append({"subject_id":str(train.subjects[global_pos]),"clean_correct":bool(correct[local]),"valid_count":int(mask.sum()),"coverage_ge2":bool(correct[local] and mask.sum()>=2),"semantic_pass_rate":float(chosen.semantic_pass[global_pos].mean())})
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,3.0); optimizer.step()
            if teacher is not None: teacher.update(net)
        print(f"[discovery] {method} f={fold} s={seed} epoch={epoch+1}",flush=True)
    frame=evaluate(net,scope,outcome,fold,seed,method,device)
    audit_frame=pd.DataFrame(final_audit)
    clean=audit_frame[audit_frame.clean_correct] if len(audit_frame) else audit_frame
    frame["coverage_ge2"]=float(clean.coverage_ge2.mean()) if len(clean) else (1.0 if not dynamic else 0.0)
    frame["median_valid_candidates"]=float(clean.valid_count.median()) if len(clean) else 0.0
    frame["semantic_pass_rate"]=float(clean.semantic_pass_rate.mean()) if len(clean) else (1.0 if not dynamic else 0.0)
    frame["bank_stability"]=float(np.mean(bank_stability)) if bank_stability else 1.0
    directory.mkdir(parents=True,exist_ok=True); c.write_csv(result_path,frame); torch.save({"state_dict":{k:v.detach().cpu() for k,v in net.state_dict().items()}},directory/"model.pt")
    del net,teacher;torch.cuda.empty_cache();return frame


def aggregate() -> dict:
    files=sorted((c.RUNTIME/"discovery_units").rglob("per_subject.csv"));frame=pd.concat([pd.read_csv(path) for path in files],ignore_index=True)
    if set(frame.method)!=set(METHODS):raise RuntimeError("DISCOVERY_METHOD_GRID_INCOMPLETE")
    c.write_csv(c.RESULTS/"DISCOVERY_PER_SUBJECT.csv",frame)
    per_fold=frame.groupby(["model","method","fold","seed"],as_index=False).agg(BA=("BA","mean"),macro_F1=("macro_F1","mean"),CE=("CE","mean"),subjects=("subject_id","nunique"));c.write_csv(c.RESULTS/"DISCOVERY_PER_FOLD.csv",per_fold)
    pivot=frame.pivot_table(index=["fold","seed","subject_id"],columns="method",values="BA").reset_index();subject=pivot.groupby("subject_id",as_index=False).mean(numeric_only=True)
    rng=np.random.default_rng(c.stable_seed("me-hard-discovery-bootstrap"));summary=[];comparisons=[]
    for method in METHODS:
        values=subject[method].to_numpy(float);draws=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1);summary.append({"method":method,"BA":float(values.mean()),"CI95_L":float(np.quantile(draws,.025)),"CI95_U":float(np.quantile(draws,.975))})
    for control in METHODS[:-1]:
        values=(subject["ME-HardSCST"]-subject[control]).to_numpy(float);draws=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1);fold_delta=pivot.groupby("fold").apply(lambda x:float((x["ME-HardSCST"]-x[control]).mean()),include_groups=False)
        comparisons.append({"comparison":f"ME-HardSCST-{control}","delta_BA":float(values.mean()),"CI95_L":float(np.quantile(draws,.025)),"CI95_U":float(np.quantile(draws,.975)),"positive_folds":int((fold_delta>0).sum())})
    c.write_csv(c.RESULTS/"DISCOVERY_SUMMARY.csv",pd.DataFrame(summary));c.write_csv(c.RESULTS/"CONTROL_COMPARISON.csv",pd.DataFrame(comparisons))
    lookup={row["comparison"]:row for row in comparisons};erm=lookup["ME-HardSCST-ERM"];hard=lookup["ME-HardSCST-Factorized-HardRandom"];uniform=lookup["ME-HardSCST-Factorized-Uniform-NoKL"]
    me=frame[frame.method=="ME-HardSCST"];source=c.read_json(c.RESULTS/"SOURCE_DECISION.json")["selected"]
    diagnostic=bool(np.isfinite(me[["coverage_ge2","semantic_pass_rate","bank_stability"]]).all().all() and me.coverage_ge2.mean()>=.5 and me.median_valid_candidates.median()>=2 and me.semantic_pass_rate.mean()>=float(source["semantic_pass_min"])-.05)
    passed=bool(erm["delta_BA"]>0 and erm["CI95_L"]>0 and erm["positive_folds"]>=3 and hard["CI95_L"]>0 and uniform["CI95_L"]>0 and diagnostic)
    stats={"bootstrap_draws":10000,"comparisons":comparisons,"diagnostic_gate":diagnostic,"discovery_supported":passed,"terminal_if_stop":None if passed else "ME_HARD_SCST_NOT_SUPPORTED","outer_or_sealed_opened":False}
    c.write_json(c.RESULTS/"STATISTICS.json",stats);return stats


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--fold",type=int,choices=c.FOLDS,required=True);parser.add_argument("--seed",type=int,choices=c.SEEDS,required=True);parser.add_argument("--method",choices=METHODS);parser.add_argument("--aggregate",action="store_true");args=parser.parse_args()
    lock=c.verify_lock(c.PROTOCOL/"ME_HARD_SCST_V2_LOCK.json");scope=str(lock["scope"]);q=float(lock["q"]);lam=float(lock["lambda_H"]);device=torch.device("cuda")
    train,sessions,outcome,_=build_cache(args.fold,args.seed,device,lock_verified=True)
    methods=METHODS if args.method is None else (args.method,)
    for method in methods:train_method(method,args.fold,args.seed,train,sessions,outcome,scope,q,lam,device)
    if args.aggregate:print(json.dumps(aggregate(),indent=2))


if __name__=="__main__":main()

