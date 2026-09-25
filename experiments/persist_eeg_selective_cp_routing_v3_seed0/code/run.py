"""UGCR-V3: frozen, utility-guided selective complement routing for EEGNet."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.nn import functional as F

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
V1_EXP = REPO / "experiments" / "persist_eeg_pc_refine_v1_seed0"
V1_CODE = V1_EXP / "code" / "run.py"
V2_EXP = REPO / "experiments" / "persist_eeg_native_cp_transfer_gate_v2_seed0"
V2_CODE = V2_EXP / "code" / "run.py"
V25_EXP = REPO / "experiments" / "persist_eeg_cp_direction_utility_v1_seed0"
V25_CODE = V25_EXP / "code" / "run.py"
os.environ.setdefault("PC_REFINE_RUNTIME", str(REPO.parent / "pc_refine_v1_runtime"))
os.environ.setdefault("PC_REFINE_ANCHORS", str(REPO / "anchors"))
os.environ.setdefault("NATIVE_GATE_RUNTIME", str(REPO.parent / "native_gate_v2_strict_runtime"))


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = import_file("ugcr_v3_native_gate_v2", V2_CODE)
B = V2.B
TASKS = tuple(B.TASKS)
FOLDS = tuple(range(5))
DEVICE = B.DEVICE
VARIANTS = ("BASELINE", "PCA_POSITIVE_ONLY_ROUTING", "SHUFFLED_UTILITY_ROUTING",
            "PCA_SIGNED_UTILITY_ROUTING", "RANDOM_BASIS_0", "RANDOM_BASIS_1",
            "RANDOM_BASIS_2", "RANDOM_BASIS_3", "RANDOM_BASIS_4")
RHO = 0.5
N_RANDOM = 5
BOOTSTRAPS = 20_000
RUNTIME = Path(os.environ.get("UGCR_V3_RUNTIME", str(REPO.parent / "ugcr_v3_seed0_runtime"))).resolve()
OUT = EXP / "outputs"
PROTOCOL = EXP / "protocol"
torch.set_num_threads(min(int(os.environ.get("UGCR_V3_CPU_THREADS", "8")), os.cpu_count() or 1))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arr_sha(*values: np.ndarray) -> str:
    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    return B.stable_seed("UGCR_V3", *parts)


def cell(task: str, fold: int) -> Path:
    return RUNTIME / "cells" / task.lower() / f"fold{fold}_seed0"


def output_cell(task: str, fold: int) -> Path:
    return OUT / "cells" / task.lower() / f"fold{fold}_seed0"


def jwrite(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def cwrite(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: row.get(k, "") for k in fields} for row in rows)
    os.replace(temp, path)


def source_commit() -> str:
    if os.environ.get("UGCR_V3_SOURCE_COMMIT"):
        return os.environ["UGCR_V3_SOURCE_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNAVAILABLE"


def v1_refit_dir(task: str, fold: int) -> Path:
    return B.RUNTIME / "refit" / task.lower() / f"fold{fold}_seed0"


def baseline_record(task: str, fold: int) -> tuple[dict, Path, dict]:
    d = v1_refit_dir(task, fold)
    rec = json.loads((d / "COMPLETE.json").read_text(encoding="utf-8"))
    ck = d / "BASELINE.pt"
    if rec.get("status") != "COMPLETE" or rec.get("task") != task or int(rec.get("fold", -1)) != fold:
        raise RuntimeError(f"canonical V1 refit checkpoint is incomplete: {task}/{fold}")
    if sha(ck) != rec.get("checkpoints", {}).get("BASELINE"):
        raise RuntimeError(f"canonical V1 checkpoint hash mismatch: {task}/{fold}")
    v2rec = json.loads((V2.cell(task, fold, "refit") / "COMPLETE.json").read_text(encoding="utf-8"))
    if v2rec.get("checkpoints", {}).get("BASELINE") != sha(ck):
        raise RuntimeError(f"V1/V2 baseline checkpoint mismatch: {task}/{fold}")
    return rec, ck, v2rec


def preflight() -> None:
    if (PROTOCOL / "V3_PROTOCOL_LOCK.json").exists():
        raise RuntimeError("protocol is already frozen")
    v2_lock = json.loads((V2_EXP / "protocol" / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    if v2_lock.get("code_sha256") != sha(V2_CODE):
        raise RuntimeError("V2 source no longer matches its protocol lock")
    v2final = V2.check_lock()
    v1_final = json.loads((V1_EXP / "protocol" / "FINAL_EVAL_LOCK.json").read_text(encoding="utf-8"))
    if sha(V1_CODE) != v1_final.get("code_sha256"):
        raise RuntimeError("canonical V1 source no longer matches its final lock")
    checks = []
    for task in TASKS:
        for fold in FOLDS:
            data_role, split, cache_name, source_sessions, future_session = B.role(task, fold)
            rec, ck, v2rec = baseline_record(task, fold)
            if rec.get("split_sha256") != split:
                raise RuntimeError(f"canonical split mismatch: {task}/{fold}")
            if rec.get("normalizer_sha256") != rec.get("normalizer", {}).get("mean_std_sha256"):
                raise RuntimeError(f"canonical normalizer record mismatch: {task}/{fold}")
            if not any(c.get("task") == task and int(c.get("fold", -1)) == fold and
                       c.get("checkpoints", {}).get("BASELINE") == sha(ck) for c in v2final["cells"]):
                raise RuntimeError(f"V2 final lock does not verify canonical baseline: {task}/{fold}")
            checks.append({"task": task, "fold": fold, "split_sha256": split,
                           "baseline_checkpoint_sha256": sha(ck),
                           "normalizer_sha256": rec["normalizer_sha256"],
                           "selected_epoch_budget": int(rec["selected_epoch_budget"]),
                           "nonfinal_refit_subject_count": len(set(map(str, data_role["inner_train_subjects"] + data_role["inner_val_subjects"] + data_role["outer_dev_subjects"]))),
                           "source_sessions": list(source_sessions), "future_session": int(future_session),
                           "v2_baseline_hash_match": True})
    OUT.mkdir(parents=True, exist_ok=True)
    cwrite(OUT / "BASELINE_CHECKPOINT_AUDIT.csv", checks)
    jwrite(RUNTIME / "PREFLIGHT.json", {"status": "PASS", "source_commit": source_commit(),
          "canonical_v1_code_sha256": sha(V1_CODE), "v2_code_sha256": sha(V2_CODE),
          "cells": len(checks), "final_heldout_eeg_reads": 0,
          "policy": "metadata/checkpoint verification only; no EEG arrays loaded"})
    print("PREFLIGHT_PASS", len(checks), "final heldout EEG array reads=0", flush=True)


def freeze_model(model) -> dict:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_identical(model, before) -> bool:
    return all(torch.equal(v, model.state_dict()[k].detach().cpu()) for k, v in before.items())


def spatial(model, x: np.ndarray, batch: int = 128) -> np.ndarray:
    values = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch])).to(DEVICE)
            hs = model.drop1(model.pool1(F.elu(model.bn2(model.spatial(model.bn1(model.temporal(xb.unsqueeze(1))))))))
            values.append(hs.flatten(1).float().cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def f0(model, hs: torch.Tensor, samples: int) -> torch.Tensor:
    shaped = hs.reshape(-1, 16, 1, samples // 4)
    return model.drop2(model.pool2(F.elu(model.bn3(model.point(model.depth(shaped)))))).flatten(1)


def successor_reconstruction_error(model, hs, qd, md, samples):
    errors=[]
    q=torch.as_tensor(qd,dtype=torch.float32,device=DEVICE);mu=torch.as_tensor(md,dtype=torch.float32,device=DEVICE)
    model.eval()
    with torch.inference_mode():
        for i in range(0,len(hs),128):
            x=torch.from_numpy(np.ascontiguousarray(hs[i:i+128])).to(DEVICE)
            y=f0(model,x,samples);centered=y-mu;p=centered@q;c=centered-p@q.T
            errors.append(torch.max(torch.abs(centered-(p@q.T+c))).item())
    return max(errors,default=0.0)


def suffix(model, hd: torch.Tensor) -> torch.Tensor:
    return model.head(model.embedding(hd))


def prepare_arrays(task: str, data: dict):
    # Read non-heldout refit roles one physical session at a time so session identity
    # is preserved for the subject-equal utility aggregation.
    subject_ids = list(map(str, data["subjects"]))
    mapping = dict(data["mapping"])
    xs=[]; ys=[]; subs=[]; sessions=[]
    for session in tuple(data["source_sessions"])+(int(data["future_session"]),):
        raw, labels, sub, mapping = B.rows(task, subject_ids, (int(session),), data["cache_name"], mapping)
        x=((raw-data["mu"][None,:,None])/np.maximum(data["sd"][None,:,None],1e-6)).astype(np.float32)
        xs.append(x);ys.append(labels.astype(np.int64));subs.append(sub.astype(str))
        sessions.append(np.full(len(labels),int(session),dtype=np.int64))
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(subs),np.concatenate(sessions)


def fit_c_basis(hs: np.ndarray, qs: np.ndarray, ms: np.ndarray, task: str, fold: int):
    centered = hs - ms[None, :]
    hc = centered - (centered @ qs) @ qs.T
    recon_error = float(np.max(np.abs(centered - (centered @ qs) @ qs.T - hc)))
    maxrank = min(16, len(hc) - 1, hc.shape[1] - qs.shape[1])
    if maxrank < 1:
        raise RuntimeError(f"empty complement rank in {task}/{fold}")
    pca = PCA(n_components=maxrank, svd_solver="randomized", random_state=stable_seed("C_PCA", task, fold))
    pca.fit(hc)
    vals = pca.singular_values_
    rank = int((vals > max(float(vals[0]) * 1e-6, 1e-8)).sum()) if len(vals) else 0
    if rank < 1:
        raise RuntimeError(f"undefined complement PCA rank: {task}/{fold}")
    u = pca.components_[:rank].T.astype(np.float64)
    q64 = qs.astype(np.float64)
    u -= q64 @ (q64.T @ u)
    u, _ = np.linalg.qr(u, mode="reduced")
    for j in range(u.shape[1]):
        pivot = int(np.argmax(np.abs(u[:, j])))
        if u[pivot, j] < 0:
            u[:, j] *= -1
    basis = u.astype(np.float32)
    coeff = hc @ basis
    ck = coeff @ basis.T
    c_residual = hc - ck
    reconstruction = float(np.max(np.abs(hc - (c_residual + ck))))
    err = reconstruction
    ortho = float(np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1]))))
    overlap = float(np.max(np.abs(qs.T @ basis)))
    if ortho >= 1e-5 or overlap >= 1e-5 or reconstruction >= 1e-5 or recon_error >= 1e-5:
        raise RuntimeError(f"complement basis audit failed: {task}/{fold} {ortho=} {overlap=} {reconstruction=}")
    record = {"status": "COMPLETE", "K_C": int(rank), "basis_sha256": arr_sha(basis),
              "pca_seed": stable_seed("C_PCA", task, fold), "singular_values": vals[:rank].tolist(),
              "explained_variance_ratio": pca.explained_variance_ratio_[:rank].tolist(),
              "orthogonality_max_abs_error": ortho, "protected_overlap_max_abs_error": overlap,
              "spatial_decomposition_max_abs_error": recon_error,
              "top_k_reconstruction_max_abs_error": reconstruction,
              "c_reconstruction_check_max_abs_error": err,
              "training_trial_count": int(len(hs)), "spatial_feature_dim": int(hs.shape[1]),
              "complement_rank_policy": "K=min(16, rank(C)); SVD singular value > max(s0*1e-6,1e-8)"}
    return basis, hc.astype(np.float32), record


def utility_from_train(model, hs, labels, subjects, sessions, qs, ms, qd, md, basis, samples, task, fold):
    """V2.5 native-coordinate erasure P-mediated margin utility, training labels only."""
    result = V2_UTILITY.eval_primary(model, hs, labels, subjects, sessions, qs, ms, qd, md,
                                     basis[None, :, :], samples, "FINAL_REFIT_TRAIN", batch_samples=128, batch_dirs=4)
    # Replicate the V2.5 hierarchy: trial means within subject-session; then equal subject means.
    subj_sessions = sorted(set(zip(result["subjects"].tolist(), result["sessions"].tolist())))
    ss_values = {}
    for sub, ses in subj_sessions:
        ix = (result["subjects"] == sub) & (result["sessions"] == ses)
        ss_values[(sub, int(ses))] = result["U_P"][ix].mean(axis=0).astype(np.float64)
    ids = sorted({s for s, _ in subj_sessions}, key=lambda s: int(str(s).replace("sub-", "")))
    subject_matrix = np.stack([np.mean([v for (s, _), v in ss_values.items() if s == sub], axis=0) for sub in ids])
    mean = subject_matrix.mean(axis=0)
    bootstrap_seed = stable_seed("SUBJECT_BOOTSTRAP", task, fold, task_key(subjects), len(ids))
    rng = np.random.default_rng(bootstrap_seed)
    boot = np.empty((BOOTSTRAPS, basis.shape[1]), dtype=np.float32)
    for start in range(0, BOOTSTRAPS, 256):
        stop = min(BOOTSTRAPS, start + 256)
        ix = rng.integers(0, len(ids), size=(stop - start, len(ids)))
        boot[start:stop] = subject_matrix[ix].mean(axis=1)
    lo = np.quantile(boot, 0.025, axis=0)
    hi = np.quantile(boot, 0.975, axis=0)
    status = np.where(lo > 0, "C_PLUS", np.where(hi < 0, "C_MINUS", "C_ZERO"))
    return result, ids, subject_matrix, mean, lo, hi, status, bootstrap_seed


def task_key(subjects):
    return "|".join(sorted(set(map(str, subjects))))


def route_coefficients(up, lo, hi, status):
    signed = np.isin(status, ("C_PLUS", "C_MINUS"))
    scale = float(np.max(np.abs(up[signed]))) if np.any(signed) else 0.0
    if not np.isfinite(scale) or scale <= 0:
        primary = np.ones(len(up), dtype=np.float32)
        positive = primary.copy()
    else:
        normed = np.clip(up / scale, -1.0, 1.0)
        signed_score = np.where(signed, normed, 0.0)
        primary = (1.0 + RHO * signed_score).astype(np.float32)
        positive = (1.0 + RHO * np.where(status == "C_PLUS", normed, 0.0)).astype(np.float32)
    return scale, positive, primary


def derangement(k: int, seed: int) -> np.ndarray:
    if k <= 1:
        return np.arange(k, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        p = rng.permutation(k)
        if np.all(p != np.arange(k)):
            return p.astype(np.int64)
    return np.roll(np.arange(k), 1).astype(np.int64)


def random_bases(qs: np.ndarray, k: int, task: str, fold: int):
    values = []
    for draw in range(N_RANDOM):
        seed = stable_seed("RANDOM_COMPLEMENT_BASIS", task, fold, draw)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((qs.shape[0], k))
        q = qs.astype(np.float64)
        z -= q @ (q.T @ z)
        u, _ = np.linalg.qr(z, mode="reduced")
        if u.shape[1] != k:
            raise RuntimeError("random spatial-complement basis has wrong rank")
        for j in range(k):
            pivot = int(np.argmax(np.abs(u[:, j])))
            if u[pivot, j] < 0:
                u[:, j] *= -1
        u = u.astype(np.float32)
        if np.max(np.abs(qs.T @ u)) >= 1e-5 or np.max(np.abs(u.T @ u - np.eye(k))) >= 1e-5:
            raise RuntimeError("random basis orthogonality audit failed")
        perm_seed = stable_seed("RANDOM_ROUTE_PERMUTATION", task, fold, draw)
        perm = np.random.default_rng(perm_seed).permutation(k).astype(np.int64)
        values.append({"basis": u, "seed": seed, "hash": arr_sha(u), "coefficient_seed": perm_seed,
                       "coefficient_permutation": perm})
    return values


def full_forward(model, hs, qd, md, samples, batch=128):
    reps=[];logits=[];pfulls=[];cfulls=[]
    q=torch.as_tensor(qd,dtype=torch.float32,device=DEVICE);mu=torch.as_tensor(md,dtype=torch.float32,device=DEVICE)
    model.eval()
    with torch.inference_mode():
        for i in range(0,len(hs),batch):
            h=torch.from_numpy(np.ascontiguousarray(hs[i:i+batch])).to(DEVICE)
            y=f0(model,h,samples);centered=y-mu;p=centered@q;c=centered-p@q.T
            z=suffix(model,y).float()
            reps.append(y.float().cpu().numpy());logits.append(z.cpu().numpy())
            pfulls.append(p.cpu().numpy());cfulls.append(c.cpu().numpy())
    return {"representation":np.concatenate(reps),"logits":np.concatenate(logits),
            "protected":np.concatenate(pfulls),"complement":np.concatenate(cfulls)}


def routed_forward(model, hs, qs, ms, basis, qd, md, coefficients, samples, batch=128, use_double=False, full_cache=None):
    """Return routed successor representation/logits plus full-forward decomposition."""
    reps=[]; logits=[]; full_reps=[]; full_logits=[]; p_moves=[]; c_errors=[]; coeff_energy=[]
    dtype = torch.float64 if use_double else torch.float32
    qs_t=torch.as_tensor(qs,dtype=dtype,device=DEVICE); ms_t=torch.as_tensor(ms,dtype=dtype,device=DEVICE)
    qd_t=torch.as_tensor(qd,dtype=dtype,device=DEVICE); md_t=torch.as_tensor(md,dtype=dtype,device=DEVICE)
    u_t=torch.as_tensor(basis,dtype=dtype,device=DEVICE); r_t=torch.as_tensor(coefficients,dtype=dtype,device=DEVICE)
    model.eval()
    with torch.inference_mode():
        for i in range(0,len(hs),batch):
            h=torch.from_numpy(np.ascontiguousarray(hs[i:i+batch])).to(DEVICE).to(dtype)
            centered=h-ms_t
            p_s=(centered@qs_t)@qs_t.T
            hc=centered-p_s
            a=hc@u_t
            c_route=(hc-a@u_t.T)+(a*r_t)@u_t.T
            hroute=(ms_t+p_s+c_route).to(torch.float32)
            hbase=h.to(torch.float32)
            if full_cache is None:
                yfull=f0(model,hbase,samples).to(dtype)
            else:
                yfull=torch.from_numpy(np.ascontiguousarray(full_cache["representation"][i:i+len(h)])).to(DEVICE).to(dtype)
            yr=f0(model,hroute,samples).to(dtype)
            pfull=(yfull-md_t)@qd_t
            cfull=(yfull-md_t)-pfull@qd_t.T
            pr=(yr-md_t)@qd_t
            hd=(md_t+pr@qd_t.T+cfull).to(torch.float32)
            z=suffix(model,hd).float()
            if full_cache is None:
                z0=suffix(model,yfull.to(torch.float32)).float()
            else:
                z0=torch.from_numpy(np.ascontiguousarray(full_cache["logits"][i:i+len(h)])).to(DEVICE).float()
            ccheck=(hd.to(dtype)-md_t)-((hd.to(dtype)-md_t)@qd_t)@qd_t.T
            c_errors.append(torch.amax(torch.abs(ccheck-cfull),dim=1).cpu().numpy())
            p_moves.append((torch.linalg.vector_norm(pr-pfull,dim=1)/torch.clamp(torch.linalg.vector_norm(pfull,dim=1),min=1e-6)).cpu().numpy())
            reps.append(hd.cpu().numpy()); logits.append(z.cpu().numpy())
            full_reps.append(yfull.to(torch.float32).cpu().numpy()); full_logits.append(z0.cpu().numpy())
            coeff_energy.append(a.square().cpu().numpy())
    return {"representation":np.concatenate(reps),"logits":np.concatenate(logits),
            "full_representation":np.concatenate(full_reps),"full_logits":np.concatenate(full_logits),
            "p_movement_relative":np.concatenate(p_moves),"c_preservation_max_abs":np.concatenate(c_errors),
            "coefficients_squared":np.concatenate(coeff_energy)}


def identity_audit(model, hs, samples, qs, ms, basis, qd, md):
    before=freeze_model(model)
    routed=routed_forward(model,hs,qs,ms,basis,qd,md,np.ones(basis.shape[1],np.float32),samples,batch=64,use_double=True)
    rd=float(np.max(np.abs(routed["representation"]-routed["full_representation"])))
    ld=float(np.max(np.abs(routed["logits"]-routed["full_logits"])))
    exact=bool(np.array_equal(routed["logits"].argmax(1),routed["full_logits"].argmax(1)))
    unchanged=state_identical(model,before)
    row={"representation_max_abs_diff":rd,"logits_max_abs_diff":ld,"prediction_exact_match":exact,
         "parameters_trainable":int(sum(p.requires_grad for p in model.parameters())),
         "frozen_state_unchanged":unchanged,"status":"PASS" if rd<1e-6 and ld<1e-6 and exact and unchanged else "FAIL_CLOSED"}
    if row["status"]!="PASS":
        raise RuntimeError(f"FAIL_CLOSED identity routing audit: {row}")
    return row


def prepare(task: str, fold: int):
    d=cell(task,fold); d.mkdir(parents=True,exist_ok=True)
    if (d/"PREPARE_COMPLETE.json").exists():
        rec=json.loads((d/"PREPARE_COMPLETE.json").read_text())
        if rec.get("status")=="COMPLETE":
            print("PREPARE_ALREADY_COMPLETE",task,fold,flush=True); return
    pref=json.loads((RUNTIME/"PREFLIGHT.json").read_text())
    if pref.get("status")!="PASS": raise RuntimeError("preflight required")
    rec,ck,v2rec=baseline_record(task,fold)
    data=B.development(task,fold,refit=True); data["refit"]=True
    if data["normalizer"]["mean_std_sha256"] != rec["normalizer_sha256"]:
        raise RuntimeError(f"final refit normalizer mismatch: {task}/{fold}")
    model=B.eegnet(data)
    payload=torch.load(ck,map_location="cpu",weights_only=False)
    model.load_state_dict(payload["state_dict"],strict=True)
    before=freeze_model(model)
    x,y,subjects,sessions=prepare_arrays(task,data)
    # Recompute final-refit geometry using the audited V2 PERSIST pathway fit on this exact pool.
    projector, project_record=V2.fit_bases(model,data,task,fold)
    if projector is None or project_record.get("status")!="COMPLETE":
        raise RuntimeError(f"final refit projector undefined: {task}/{fold}")
    qs,qd,ms,md=(projector[k].astype(np.float32) for k in ("qs","qd","ms","md"))
    e_s=float(np.max(np.abs(qs.T@qs-np.eye(qs.shape[1]))))
    e_d=float(np.max(np.abs(qd.T@qd-np.eye(qd.shape[1]))))
    if e_s>=1e-5 or e_d>=1e-5: raise RuntimeError("projector orthonormality failed")
    hs=spatial(model,x)
    successor_error=successor_reconstruction_error(model,hs,qd,md,x.shape[2])
    if successor_error>=1e-5: raise RuntimeError(f"successor P/C reconstruction failed: {successor_error}")
    basis,hc,cbrec=fit_c_basis(hs,qs,ms,task,fold)
    # Exact TRAIN-only native erasure estimand copied from the audited V2.5 implementation.
    utility,ids,subject_matrix,up,lo,hi,status,bootstrap_seed=utility_from_train(model,hs,y,subjects,sessions,qs,ms,qd,md,basis,x.shape[2],task,fold)
    scale,rpos,rprimary=route_coefficients(up,lo,hi,status)
    permutation=derangement(len(rprimary),stable_seed("SHUFFLE",task,fold))
    rshuffle=rprimary[permutation]
    randoms=random_bases(qs,len(basis.T),task,fold)
    for item in randoms:
        item["coefficients"]=(1.0+(rprimary-1.0)[item["coefficient_permutation"]]).astype(np.float32)
    # Identity audit is run before the protocol lock and only on legal refit TRAIN activations.
    ident=identity_audit(model,hs[:min(64,len(hs))],x.shape[2],qs,ms,basis,qd,md)
    if not state_identical(model,before): raise RuntimeError("frozen baseline state changed during preparation")
    direction=[]; routes=[]
    for j in range(len(up)):
        direction.append({"task":task,"fold":fold,"direction":j,"U_P_subject_equal_mean":float(up[j]),
            "CI95_lower":float(lo[j]),"CI95_upper":float(hi[j]),"biological_subjects":len(ids),
            "bootstrap_draws":BOOTSTRAPS,"bootstrap_seed":bootstrap_seed,
            "class":str(status[j]),"PCA_explained_variance_fraction":float(cbrec["explained_variance_ratio"][j]),
            "primary_r":float(rprimary[j]),"positive_only_r":float(rpos[j]),"utility_scale_S":scale})
        routes.append({"task":task,"fold":fold,"direction":j,"class":str(status[j]),"U_P":float(up[j]),
            "primary_r":float(rprimary[j]),"positive_only_r":float(rpos[j]),"shuffle_source_direction":int(permutation[j]),
            "shuffled_r":float(rshuffle[j]),"variance_fraction":float(cbrec["explained_variance_ratio"][j])})
    save={"qs":qs,"qd":qd,"ms":ms,"md":md,"c_basis":basis,
          "random_bases":np.stack([q["basis"] for q in randoms]),
          "random_coefficients":np.stack([q["coefficients"] for q in randoms]),
          "r_primary":rprimary,"r_positive":rpos,"r_shuffle":rshuffle,"shuffle_permutation":permutation,
          "random_permutations":np.stack([q["coefficient_permutation"] for q in randoms]),
          "train_subject_utility":subject_matrix,"U_P":up,"CI_lower":lo,"CI_upper":hi,
          "C_class":status.astype("U"),"explained_variance_fraction":np.asarray(cbrec["explained_variance_ratio"],np.float64),
          "normalizer_mu":data["mu"],"normalizer_sd":data["sd"]}
    temp=d/"geometry.npz.part"
    with temp.open("wb") as f: np.savez_compressed(f,**save)
    os.replace(temp,d/"geometry.npz")
    jwrite(d/"PROJECTOR_AUDIT.json",{"task":task,"fold":fold,"projector_record":project_record,
        "projector_file_sha256":sha(d/"geometry.npz"),"qs_hash":arr_sha(qs,ms),"qd_hash":arr_sha(qd,md),
        "spatial_rank":int(qs.shape[1]),"successor_rank":int(qd.shape[1]),"qs_orthogonality_error":e_s,
        "qd_orthogonality_error":e_d,"spatial_decomposition_reconstruction_max_abs_error":cbrec["spatial_decomposition_max_abs_error"],
        "successor_decomposition_reconstruction_max_abs_error":successor_error,
        "training_subject_count":len(set(subjects)),"training_trial_count":len(x),
        "final_refit_normalizer_sha256":data["normalizer"]["mean_std_sha256"],
        "baseline_checkpoint_sha256":sha(ck),"baseline_state_parameter_count":sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count":sum(p.requires_grad for p in model.parameters())})
    jwrite(d/"C_BASIS_AUDIT.json",cbrec)
    cwrite(d/"DIRECTION_UTILITY.csv",direction)
    cwrite(d/"ROUTING_COEFFICIENTS.csv",routes)
    cwrite(d/"IDENTITY_ROUTING_AUDIT.csv",[{"task":task,"fold":fold,"n_train_trials":min(64,len(hs)),**ident}])
    random_audit=[]
    for draw,item in enumerate(randoms):
        for j in range(len(up)):
            random_audit.append({"task":task,"fold":fold,"draw":draw,"direction":j,"basis_seed":item["seed"],
                "basis_sha256":item["hash"],"coefficient_permutation_seed":item["coefficient_seed"],
                "coefficient_source_direction":int(item["coefficient_permutation"][j]),
                "r":float(item["coefficients"][j]),"deviation":float(item["coefficients"][j]-1.0),
                "rank":int(len(up)),"orthogonality_error":float(np.max(np.abs(item["basis"].T@item["basis"]-np.eye(len(up))))),
                "protected_overlap_error":float(np.max(np.abs(qs.T@item["basis"])))})
    cwrite(d/"RANDOM_BASIS_AUDIT.csv",random_audit)
    jwrite(d/"PREPARE_COMPLETE.json",{"status":"COMPLETE","task":task,"fold":fold,
        "baseline_checkpoint_sha256":sha(ck),"v2_baseline_checkpoint_sha256":v2rec["checkpoints"]["BASELINE"],
        "normalizer_sha256":data["normalizer"]["mean_std_sha256"],"split_sha256":data["split"],
        "training_pool_role":"inner_train + inner_val + outer_dev; final-heldout excluded",
        "training_subjects":len(set(subjects)),"training_trials":len(x),"geometry_sha256":sha(d/"geometry.npz"),
        "projector_audit_sha256":sha(d/"PROJECTOR_AUDIT.json"),"c_basis_audit_sha256":sha(d/"C_BASIS_AUDIT.json"),
        "direction_utility_sha256":sha(d/"DIRECTION_UTILITY.csv"),"identity_audit_sha256":sha(d/"IDENTITY_ROUTING_AUDIT.csv"),
        "final_heldout_array_reads":0,"model_frozen":True,"new_trainable_parameters":0})
    print("PREPARE_COMPLETE",task,fold,"trials",len(x),"subjects",len(ids),"K",len(up),
          "C+",int(np.sum(status=="C_PLUS")),"C-",int(np.sum(status=="C_MINUS")),flush=True)


def load_v25_utility():
    module=import_file("ugcr_v3_v25_utility",V25_CODE)
    if module.sha(V25_CODE) != json.loads((V25_EXP/"protocol"/"PROTOCOL_LOCK.json").read_text())["code_sha256"]:
        raise RuntimeError("V2.5 utility source no longer matches its frozen protocol")
    return module


V2_UTILITY=load_v25_utility()


def lock_protocol():
    lock_path=PROTOCOL/"V3_PROTOCOL_LOCK.json"
    if lock_path.exists(): raise RuntimeError("V3 protocol lock already exists; immutable")
    cells=[]; projector_rows=[]; basis_rows=[]; utility_rows=[]; route_rows=[]; shuffle_rows=[]; random_rows=[]; identity_rows=[]
    for task in TASKS:
        for fold in FOLDS:
            d=cell(task,fold); comp=json.loads((d/"PREPARE_COMPLETE.json").read_text(encoding="utf-8"))
            if comp.get("status")!="COMPLETE": raise RuntimeError(f"preparation incomplete: {task}/{fold}")
            if sha(d/"geometry.npz")!=comp["geometry_sha256"]: raise RuntimeError("geometry changed before lock")
            ge=np.load(d/"geometry.npz",allow_pickle=False)
            proj=json.loads((d/"PROJECTOR_AUDIT.json").read_text(encoding="utf-8")); cb=json.loads((d/"C_BASIS_AUDIT.json").read_text(encoding="utf-8"))
            urows=list(csv.DictReader((d/"DIRECTION_UTILITY.csv").open(newline="",encoding="utf-8")))
            rrows=list(csv.DictReader((d/"ROUTING_COEFFICIENTS.csv").open(newline="",encoding="utf-8")))
            brow={k:ge[k] for k in ge.files}
            rec,ck,v2rec=baseline_record(task,fold)
            idrows=list(csv.DictReader((d/"IDENTITY_ROUTING_AUDIT.csv").open(newline="",encoding="utf-8")))
            if idrows[0]["status"]!="PASS": raise RuntimeError("identity audit failed; refusing protocol lock")
            projector_rows.append({"task":task,"fold":fold,"projector_sha256":sha(d/"geometry.npz"),**{k:proj[k] for k in ("qs_hash","qd_hash","spatial_rank","successor_rank","qs_orthogonality_error","qd_orthogonality_error","spatial_decomposition_reconstruction_max_abs_error","successor_decomposition_reconstruction_max_abs_error","final_refit_normalizer_sha256")}})
            basis_rows.append({"task":task,"fold":fold,"C_basis_sha256":arr_sha(brow["c_basis"]),"K_C":int(brow["c_basis"].shape[1]),
                "explained_variance_fraction":brow["explained_variance_fraction"].tolist(),"fit_subject_count":comp["training_subjects"],
                "fit_trial_count":comp["training_trials"],"reconstruction_error":cb["top_k_reconstruction_max_abs_error"]})
            utility_rows.extend([{k:(int(v) if k in ("fold","direction","biological_subjects","bootstrap_draws") else float(v) if k in ("U_P_subject_equal_mean","CI95_lower","CI95_upper","PCA_explained_variance_fraction","primary_r","positive_only_r","utility_scale_S") else v) for k,v in row.items()} for row in urows])
            route_rows.extend([{**r,"fold":int(r["fold"]),"direction":int(r["direction"]),"primary_r":float(r["primary_r"]),"positive_only_r":float(r["positive_only_r"]),"shuffled_r":float(r["shuffled_r"])} for r in rrows])
            perm=brow["shuffle_permutation"].astype(int).tolist()
            shuffle_rows.extend({"task":task,"fold":fold,"target_direction":j,"source_direction":int(perm[j]),
                "target_r":float(brow["r_shuffle"][j]),"source_r":float(brow["r_primary"][perm[j]]),
                "basis_sha256":arr_sha(brow["c_basis"]),"derangement":bool(perm[j]!=j)} for j in range(len(perm)))
            randommeta=[]
            for draw in range(N_RANDOM):
                rb=brow["random_bases"][draw]; rp=brow["random_permutations"][draw].astype(int)
                info={"draw":draw,"basis_sha256":arr_sha(rb),"coefficient_permutation":rp.tolist(),
                      "coefficients":brow["random_coefficients"][draw].astype(float).tolist(),
                      "rank":int(rb.shape[1]),"deviation_l2":float(np.linalg.norm(brow["random_coefficients"][draw]-1)),
                      "basis_seed":stable_seed("RANDOM_COMPLEMENT_BASIS",task,fold,draw),
                      "permutation_seed":stable_seed("RANDOM_ROUTE_PERMUTATION",task,fold,draw)}
                randommeta.append(info)
                random_rows.extend({"task":task,"fold":fold,**{k:v for k,v in info.items() if k not in ("coefficients","coefficient_permutation")},
                    "direction":j,"coefficient_source_direction":int(rp[j]),"r":float(brow["random_coefficients"][draw,j]),
                    "deviation":float(brow["random_coefficients"][draw,j]-1),
                    "same_primary_deviation_multiset":bool(np.allclose(np.sort(brow["random_coefficients"][draw]-1),np.sort(brow["r_primary"]-1),rtol=0,atol=1e-7)),
                    "orthogonality_error":float(np.max(np.abs(rb.T@rb-np.eye(rb.shape[1])))),
                    "protected_overlap_error":float(np.max(np.abs(brow["qs"].T@rb)))} for j in range(len(rp)))
            identity_rows.extend(idrows)
            cells.append({"task":task,"fold":fold,"baseline_checkpoint_sha256":sha(ck),"v1_complete_sha256":sha(v1_refit_dir(task,fold)/"COMPLETE.json"),
                "v2_complete_sha256":sha(V2.cell(task,fold,"refit")/"COMPLETE.json"),"split_sha256":comp["split_sha256"],
                "normalizer_sha256":comp["normalizer_sha256"],"geometry_sha256":comp["geometry_sha256"],
                "qs_hash":proj["qs_hash"],"qd_hash":proj["qd_hash"],"c_basis_hash":arr_sha(brow["c_basis"]),
                "c_basis_rank":int(brow["c_basis"].shape[1]),"U_P":brow["U_P"].astype(float).tolist(),
                "U_P_CI95_lower":brow["CI_lower"].astype(float).tolist(),"U_P_CI95_upper":brow["CI_upper"].astype(float).tolist(),
                "C_class":brow["C_class"].astype(str).tolist(),"r_primary":brow["r_primary"].astype(float).tolist(),
                "r_positive_only":brow["r_positive"].astype(float).tolist(),"shuffle_permutation":perm,"random_bases":randommeta,
                "identity_audit_sha256":sha(d/"IDENTITY_ROUTING_AUDIT.csv"),"geometry_file_sha256":sha(d/"geometry.npz")})
            ge.close()
    source_files={str(p.relative_to(REPO)).replace("\\","/"):sha(p) for p in sorted((EXP/"code").glob("*.py"))}
    deps={str(p.relative_to(REPO)).replace("\\","/"):sha(p) for p in (V1_CODE,V2_CODE,V25_CODE,
        B.SEVEN_CODE/"backbone_models.py", B.SEVEN_CODE/"tech_recipe_selection.py",
        REPO/"experiments"/"persist_eeg_crossbackbone_peeh_v1"/"code"/"run_crossbackbone_peeh.py")}
    h=hashlib.sha256(json.dumps({"sources":source_files,"dependencies":deps},sort_keys=True).encode()).hexdigest()
    manifest=json.loads((REPO/"experiments"/"persist_eeg_final_heldout_confirmation_v1"/"protocol"/"FINAL_HOLDOUT_MANIFEST.json").read_text(encoding="utf-8"))
    open_ids=sorted(map(str,manifest["OpenBMI"]["subject_ids"]),key=lambda s:int(s.replace("sub-","")))
    if len(open_ids)!=14 or len(B.TRUE_WBCIC)!=10: raise RuntimeError("formal heldout metadata count mismatch")
    # Protocol-only artifacts remain TRAIN-derived; no heldout EEG arrays are loaded here.
    cwrite(OUT/"BASELINE_CHECKPOINT_AUDIT.csv",[{"task":r["task"],"fold":r["fold"],"baseline_checkpoint_sha256":r["baseline_checkpoint_sha256"],
        "normalizer_sha256":r["normalizer_sha256"],"split_sha256":r["split_sha256"],"V1_V2_hash_match":True,"new_trainable_parameters":0,
        "model_state":"all parameters and BN buffers frozen"} for r in cells])
    cwrite(OUT/"FINAL_REFIT_PROJECTOR_AUDIT.csv",projector_rows)
    cwrite(OUT/"FINAL_REFIT_C_BASIS_AUDIT.csv",basis_rows)
    cwrite(OUT/"FINAL_REFIT_DIRECTION_UTILITY.csv",utility_rows)
    cwrite(OUT/"ROUTING_COEFFICIENTS.csv",route_rows)
    cwrite(OUT/"SHUFFLE_CONTROL_AUDIT.csv",shuffle_rows)
    cwrite(OUT/"RANDOM_BASIS_CONTROL_AUDIT.csv",random_rows)
    cwrite(OUT/"IDENTITY_ROUTING_AUDIT.csv",identity_rows)
    # Copy immutable small TRAIN-derived geometry needed by the evaluator into versioned outputs.
    for task in TASKS:
        for fold in FOLDS:
            dst=output_cell(task,fold);dst.mkdir(parents=True,exist_ok=True)
            src=cell(task,fold)
            for filename in ("geometry.npz","PROJECTOR_AUDIT.json","C_BASIS_AUDIT.json","PREPARE_COMPLETE.json"):
                (dst/filename).write_bytes((src/filename).read_bytes())
    code_hash=hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest()
    lock={"schema":"PERSIST_EEG_SELECTIVE_CP_ROUTING_V3_SEED0","created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "source_commit":source_commit(),"code_hash_sha256":code_hash,"evaluation_code_and_dependency_hash_sha256":h,
        "code_files":source_files,"dependency_hashes":deps,"scope":{"tasks":list(TASKS),"folds":list(FOLDS),"seed":0,"backbone":"EEGNet"},
        "analysis_status":"POST_HELDOUT_DEVELOPMENT_EVALUATION","heldout_accessed_before_lock":False,
        "baseline":{"source":"exact canonical V1/V2 EEGNet final-refit checkpoint","all_parameters_and_bn_buffers_frozen":True,"new_trainable_parameters":0,"cells":cells},
        "geometry":{"procedure":"recomputed with audited V2 PERSIST final-refit fit_bases procedure on non-final-heldout refit pool",
            "transition":"spatial_elu_pool1 -> depth_point_elu_pool2","projectors":projector_rows,"C_basis":basis_rows,
            "PCA":"randomized TRAIN-only PCA on h_C; K=min(16,rank(C)); canonical signs; seed locked per task/fold"},
        "utility":{"source":"exact V2.5 eval_primary native-coordinate erasure P-mediated margin estimand",
            "formula":"U_P=m(z0,y)-m(z_minus_j_P,y); train/refit labels only","aggregation":"session means within biological subject, then subject-equal",
            "bootstrap":{"unit":"biological subject","draws":BOOTSTRAPS,"CI":"percentile 95%"},"directions":utility_rows},
        "routing":{"rho":RHO,"scale":"S=max |U_P| across significant C+ and C- directions; r=1 when none",
            "primary":"r_j=1+rho*clip(U_P/S,-1,1) for C+/C-; C0=1","positive_only":"boost C+ with same S, other directions r=1",
            "shuffle":"same PCA basis and exact primary coefficient multiset; prelocked derangement","primary_coefficients":route_rows,
            "shuffle_mapping":shuffle_rows,"random_basis_controls":random_rows},
        "variants":list(VARIANTS),"random_basis_draws":N_RANDOM,
        "model_construction":"h_s_route=mu_s+Q_s Q_s^T(h_s-mu_s)+C_residual+sum_j r_j a_j c_j; preserve native full-forward successor C_d in h_d=mu_d+Q_d p_d_route+c_d_full",
        "identity_routing_audit":identity_rows,"heldout_cohorts":{"OpenBMI_final_heldout_subjects":open_ids,"WBCIC_true_outer_subjects":list(B.TRUE_WBCIC)},
        "evaluation":{"metric_primary":"biological-subject-equal future-session S2 balanced accuracy after five-fold probability mean",
            "sessions":{"OpenBMI_MI_ERP_SSVEP":[1,2],"WBCIC_MI":[0,1,2]},"secondary":["subject-equal macro-F1","subject-equal worst-session BA","subject-equal future-session NLL","per-session BA","subject-level BA"],
            "probability_aggregation":"softmax each fold logits, average probabilities equally across five folds",
            "paired_bootstrap":{"unit":"biological subject within task","draws":BOOTSTRAPS,"comparisons":["UGCR-V3 vs baseline","UGCR-V3 vs positive-only","UGCR-V3 vs shuffled","UGCR-V3 vs random mean","UGCR-V3 vs best random draw"]}},
        "interpretation_state_rules":{"SELECTIVE_ROUTING_ACTIONABLE":"UGCR-V3 BA CI lower > 0 vs baseline, shuffled, random mean, and best random",
            "POSITIVE_ROUTING_ONLY":"positive-only minus baseline BA CI lower > 0, while UGCR-V3 minus positive-only BA CI upper <= 0",
            "UTILITY_NOT_DIRECTION_SPECIFIC":"UGCR-V3 minus shuffled and random-mean BA CIs both include 0",
            "UTILITY_STRUCTURE_NON_ACTIONABLE":"UGCR-V3 minus baseline and positive-only minus baseline BA CI upper <= 0",
            "TASK_DEPENDENT_SELECTIVE_ROUTING":"other task-specific or mixed result patterns"},
        "no_training":True,"protocol_freeze":"no routing, basis, coefficient, threshold, metric, or evaluation-code changes after this file is written",
        "heldout_array_reads_at_lock":0}
    jwrite(lock_path,lock)
    (PROTOCOL/"V3_PROTOCOL_LOCK.sha256").write_text(sha(lock_path)+"\n",encoding="utf-8")
    print("V3_PROTOCOL_LOCKED",sha(lock_path),"heldout EEG reads=0",flush=True)


def verify_protocol():
    path=PROTOCOL/"V3_PROTOCOL_LOCK.json";digest=PROTOCOL/"V3_PROTOCOL_LOCK.sha256"
    if not path.is_file() or not digest.is_file() or sha(path)!=digest.read_text(encoding="utf-8").strip():
        raise RuntimeError("V3 protocol lock missing or hash mismatch")
    rec=json.loads(path.read_text(encoding="utf-8"))
    if rec.get("analysis_status")!="POST_HELDOUT_DEVELOPMENT_EVALUATION" or rec.get("heldout_accessed_before_lock") is not False:
        raise RuntimeError("protocol status or pre-lock heldout policy mismatch")
    for rel,h in rec["code_files"].items():
        if sha(REPO/rel)!=h: raise RuntimeError(f"code changed after protocol lock: {rel}")
    for rel,h in rec["dependency_hashes"].items():
        if sha(REPO/rel)!=h: raise RuntimeError(f"dependency changed after protocol lock: {rel}")
    for c in rec["baseline"]["cells"]:
        d=output_cell(c["task"],int(c["fold"]))
        if sha(d/"geometry.npz")!=c["geometry_file_sha256"]: raise RuntimeError("locked geometry changed")
    return rec


def diagnostic_features(hs, qs, ms, basis, cplus, cminus, highdirs):
    centered=hs-ms[None,:]; hc=centered-(centered@qs)@qs.T; a=hc@basis
    denom=np.maximum(np.sum(hc*hc,axis=1),1e-12)
    pos=np.sum(a[:,cplus]**2,axis=1) if np.any(cplus) else np.zeros(len(hs))
    neg=np.sum(a[:,cminus]**2,axis=1) if np.any(cminus) else np.zeros(len(hs))
    high=np.sum(a[:,highdirs]**2,axis=1) if np.any(highdirs) else np.zeros(len(hs))
    return np.stack((pos/denom,neg/denom,high/denom),axis=1).astype(np.float32)


def final_eval(task: str, fold: int):
    lock=verify_protocol()  # The only path to a formal heldout EEG array read.
    target=output_cell(task,fold)/"heldout_predictions.npz"
    if target.exists():
        raise RuntimeError(f"heldout cell already exists; refusing overwrite: {task}/{fold}")
    cell_lock=next(c for c in lock["baseline"]["cells"] if c["task"]==task and int(c["fold"])==fold)
    geom=np.load(output_cell(task,fold)/"geometry.npz",allow_pickle=False)
    # Recreate exact nonfinal normalization/map from the refit pool; these remain TRAIN/DEV arrays.
    data=B.development(task,fold,refit=True)
    if data["normalizer"]["mean_std_sha256"]!=cell_lock["normalizer_sha256"]: raise RuntimeError("refit normalizer drift")
    baseline,ck,v2rec=baseline_record(task,fold)
    if sha(ck)!=cell_lock["baseline_checkpoint_sha256"]: raise RuntimeError("baseline checkpoint drift")
    model=B.eegnet(data); model.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["state_dict"],strict=True)
    before=freeze_model(model)
    held=lock["heldout_cohorts"]["WBCIC_true_outer_subjects" if task=="WBCIC_MI" else "OpenBMI_final_heldout_subjects"]
    if set(held)&set(map(str,data["subjects"])): raise RuntimeError("development/final-heldout overlap")
    sessions=(0,1,2) if task=="WBCIC_MI" else (1,2)
    qs,qd,ms,md,basis=(geom[k] for k in ("qs","qd","ms","md","c_basis"))
    rbase=np.ones(len(basis.T),np.float32)
    rpos=geom["r_positive"];rshuffle=geom["r_shuffle"];rprim=geom["r_primary"]
    variants={"PCA_POSITIVE_ONLY_ROUTING":(basis,rpos),"SHUFFLED_UTILITY_ROUTING":(basis,rshuffle),
              "PCA_SIGNED_UTILITY_ROUTING":(basis,rprim)}
    for draw in range(N_RANDOM): variants[f"RANDOM_BASIS_{draw}"]=(geom["random_bases"][draw],geom["random_coefficients"][draw])
    output={}; mov_rows=[]; preserve_max=0.0
    means=geom["normalizer_mu"];sds=geom["normalizer_sd"]
    cclass=geom["C_class"].astype(str); cplus=(cclass=="C_PLUS");cminus=(cclass=="C_MINUS")
    absu=np.abs(geom["U_P"]); high=absu>=np.quantile(absu,.75) if len(absu) else np.zeros(0,bool)
    for session in sessions:
        raw,y,subjects,_=B.rows(task,held,(session,),data["cache_name"],data["mapping"],final=(task=="WBCIC_MI"))
        x=((raw-means[None,:,None])/np.maximum(sds[None,:,None],1e-6)).astype(np.float32)
        del raw
        hs=spatial(model,x)
        diag=diagnostic_features(hs,qs,ms,basis,cplus,cminus,high)
        # The common baseline full pathway is evaluated once. Every comparator adds only a routed F0 path.
        baseout=full_forward(model,hs,qd,md,x.shape[2],batch=128)
        output[f"S{session}_y"]=y.astype(np.int64);output[f"S{session}_subjects"]=subjects.astype("U")
        output[f"BASELINE_S{session}_z"]=baseout["logits"].astype(np.float32)
        output[f"S{session}_diag"]=diag
        output[f"BASELINE_S{session}_p_movement_relative"]=np.zeros(len(x),np.float32)
        output[f"BASELINE_S{session}_c_preservation_max_abs"]=np.zeros(len(x),np.float32)
        for name,(u,r) in variants.items():
            out=routed_forward(model,hs,qs,ms,u,qd,md,r,x.shape[2],batch=128,full_cache=baseout)
            output[f"{name}_S{session}_z"]=out["logits"].astype(np.float32)
            output[f"{name}_S{session}_p_movement_relative"]=out["p_movement_relative"].astype(np.float32)
            output[f"{name}_S{session}_c_preservation_max_abs"]=out["c_preservation_max_abs"].astype(np.float32)
            preserve_max=max(preserve_max,float(np.max(out["c_preservation_max_abs"])))
            mov_rows.append({"task":task,"fold":fold,"session":f"S{session}","variant":name,
                "mean_relative_P_movement":float(out["p_movement_relative"].mean()),
                "median_relative_P_movement":float(np.median(out["p_movement_relative"])),
                "max_C_preservation_abs_error":float(out["c_preservation_max_abs"].max()),
                "label":"POST_HOC_HELDOUT_DIAGNOSTIC"})
        if not state_identical(model,before): raise RuntimeError("frozen model state changed during heldout evaluation")
        print("HELDOUT_SESSION_COMPLETE",task,fold,session,len(x),flush=True)
        del x,hs
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix(".npz.part")
    with temp.open("wb") as f: np.savez_compressed(f,**output)
    os.replace(temp,target)
    cwrite(output_cell(task,fold)/"HELDOUT_MOVEMENT_AUDIT.csv",mov_rows)
    jwrite(output_cell(task,fold)/"HELDOUT_COMPLETE.json",{"status":"COMPLETE","task":task,"fold":fold,
        "prediction_sha256":sha(target),"movement_audit_sha256":sha(output_cell(task,fold)/"HELDOUT_MOVEMENT_AUDIT.csv"),
        "protocol_lock_sha256":sha(PROTOCOL/"V3_PROTOCOL_LOCK.json"),"outcome_scope":"POST_HELDOUT_DEVELOPMENT_EVALUATION",
        "heldout_subjects":len(set(held)),"sessions":list(sessions),"preserve_full_native_C":True,
        "max_C_preservation_abs_error":preserve_max,"model_state_unchanged":True})
    geom.close()


def softmax(z):
    zz=z-z.max(axis=-1,keepdims=True);e=np.exp(zz);return e/e.sum(axis=-1,keepdims=True)


def paired_bootstrap(a: np.ndarray,b: np.ndarray,seed: int):
    d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
    rng=np.random.default_rng(seed); boot=np.empty(BOOTSTRAPS,np.float32)
    for start in range(0,BOOTSTRAPS,512):
        stop=min(BOOTSTRAPS,start+512);ix=rng.integers(0,len(d),(stop-start,len(d)));boot[start:stop]=d[ix].mean(axis=1)
    return float(d.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975))


def aggregate():
    lock=verify_protocol(); subjects_out=[]; summary=[]; contrasts=[]; rescue=[]; movement=[]; session_rows=[]
    for task in TASKS:
        parts=[]
        for fold in FOLDS:
            d=output_cell(task,fold); complete=json.loads((d/"HELDOUT_COMPLETE.json").read_text(encoding="utf-8"))
            path=d/"heldout_predictions.npz"
            if complete.get("status")!="COMPLETE" or sha(path)!=complete["prediction_sha256"]: raise RuntimeError("heldout cell output hash mismatch")
            parts.append(np.load(path,allow_pickle=False))
            movement.extend(list(csv.DictReader((d/"HELDOUT_MOVEMENT_AUDIT.csv").open(newline="",encoding="utf-8"))))
        sessions=(0,1,2) if task=="WBCIC_MI" else (1,2); by={}
        for session in sessions:
            y=parts[0][f"S{session}_y"]; subs=parts[0][f"S{session}_subjects"].astype(str)
            for p in parts[1:]:
                if not np.array_equal(y,p[f"S{session}_y"]) or not np.array_equal(subs,p[f"S{session}_subjects"].astype(str)):
                    raise RuntimeError("heldout trial order differs between fold predictions")
            probabilities={}; logits={}
            for v in VARIANTS:
                zs=np.stack([p[f"{v}_S{session}_z"] for p in parts]); pr=softmax(zs).mean(axis=0)
                probabilities[v]=pr; logits[v]=zs
            for sub in sorted(set(subs),key=lambda x:int(x.replace("sub-",""))):
                ix=subs==sub
                for v in VARIANTS:
                    pr=probabilities[v][ix];pred=pr.argmax(1)
                    row={"task":task,"subject":sub,"session":f"S{session}","variant":v,
                         "BA":float(balanced_accuracy_score(y[ix],pred)),
                         "macro_F1":float(f1_score(y[ix],pred,average="macro",zero_division=0)),
                         "NLL":float(-np.log(np.clip(pr[np.arange(len(pr)),y[ix]],1e-12,1)).mean()),"trials":int(ix.sum()),
                         "label":"POST_HELDOUT_DEVELOPMENT_EVALUATION"}
                    subjects_out.append(row);by[(sub,session,v)]=row
                for v in VARIANTS[1:]:
                    bp=probabilities["BASELINE"][ix].argmax(1);vp=probabilities[v][ix].argmax(1)
                    rescued=(bp!=y[ix])&(vp==y[ix]);damaged=(bp==y[ix])&(vp!=y[ix])
                    diag=np.mean([p[f"S{session}_diag"][ix] for p in parts],axis=0)
                    entry={"task":task,"subject":sub,"session":f"S{session}","variant":v,
                        "baseline_errors_rescued":int(rescued.sum()),"baseline_correct_damaged":int(damaged.sum()),
                        "trials":int(ix.sum()),"label":"POST_HOC_HELDOUT_DIAGNOSTIC"}
                    for k,name in enumerate(("Cplus_energy_fraction","Cminus_energy_fraction","high_abs_utility_energy_fraction")):
                        entry[name+"_rescued_mean"]=float(diag[rescued,k].mean()) if rescued.any() else ""
                        entry[name+"_damaged_mean"]=float(diag[damaged,k].mean()) if damaged.any() else ""
                    # Route movement averaged across folds and limited to the exact session/subset.
                    for key in ("p_movement_relative","c_preservation_max_abs"):
                        vals=np.mean([p[f"{v}_S{session}_{key}"][ix] for p in parts],axis=0)
                        entry[key+"_rescued_mean"]=float(vals[rescued].mean()) if rescued.any() else ""
                        entry[key+"_damaged_mean"]=float(vals[damaged].mean()) if damaged.any() else ""
                    rescue.append(entry)
            for v in VARIANTS:
                rr=[by[(s,session,v)] for s in set(subs)]
                session_rows.append({"task":task,"session":f"S{session}","variant":v,"subjects":len(rr),
                    **{m:float(np.mean([r[m] for r in rr])) for m in ("BA","macro_F1","NLL")},
                    "label":"POST_HELDOUT_DEVELOPMENT_EVALUATION"})
        hids=sorted({s for s,_,_ in by},key=lambda x:int(x.replace("sub-","")))
        if len(hids)!=(10 if task=="WBCIC_MI" else 14): raise RuntimeError("heldout subject count mismatch")
        vectors={}
        for v in VARIANTS:
            for m in ("BA","macro_F1","NLL"):
                vectors[(v,m)]=np.asarray([by[(s,2,v)][m] for s in hids])
            vectors[(v,"worst_session_BA")]=np.asarray([min(by[(s,se,v)]["BA"] for se in sessions) for s in hids])
            summary.append({"task":task,"variant":v,"subjects":len(hids),"primary_session":"S2",
                **{m:float(vectors[(v,m)].mean()) for m in ("BA","macro_F1","NLL","worst_session_BA")},
                "label":"POST_HELDOUT_DEVELOPMENT_EVALUATION"})
        random_names=[f"RANDOM_BASIS_{i}" for i in range(N_RANDOM)]
        random_best=max(random_names,key=lambda name:float(vectors[(name,"BA")].mean()))
        random_worst=min(random_names,key=lambda name:float(vectors[(name,"BA")].mean()))
        for name, draws in (("RANDOM_BASIS_MEAN",random_names),("RANDOM_BASIS_BEST_BA",[random_best]),
                            ("RANDOM_BASIS_WORST_BA",[random_worst])):
            averaged={metric:np.mean([vectors[(draw,metric)] for draw in draws],axis=0)
                      for metric in ("BA","macro_F1","NLL","worst_session_BA")}
            summary.append({"task":task,"variant":name,"subjects":len(hids),"primary_session":"S2",
                **{metric:float(value.mean()) for metric,value in averaged.items()},
                "random_draws":len(draws),"label":"POST_HELDOUT_DEVELOPMENT_EVALUATION"})
        comps=[("BASELINE","BASELINE"),("PCA_POSITIVE_ONLY_ROUTING","POSITIVE_ONLY"),
            ("SHUFFLED_UTILITY_ROUTING","SHUFFLED"),("RANDOM_MEAN","RANDOM_MEAN"),(random_best,"BEST_RANDOM")]
        for metric in ("BA","macro_F1","worst_session_BA","NLL"):
            a=vectors[("PCA_SIGNED_UTILITY_ROUTING",metric)]
            for comp,label in comps:
                if comp=="RANDOM_MEAN":b=np.mean([vectors[(r,metric)] for r in random_names],axis=0)
                else:b=vectors[(comp,metric)]
                delta,lo,hi=paired_bootstrap(a,b,stable_seed("PAIRED_BOOTSTRAP",task,label,metric))
                contrasts.append({"task":task,"contrast":"UGCR_V3 - "+label,"comparator_variant":comp,
                    "metric":metric,"difference":delta,"CI95_low":lo,"CI95_high":hi,
                    "bootstrap_draws":BOOTSTRAPS,"unit":"biological subject","selected_best_random_draw":random_best,
                    "label":"POST_HELDOUT_DEVELOPMENT_EVALUATION"})
            if metric in ("BA","macro_F1","worst_session_BA","NLL"):
                delta,lo,hi=paired_bootstrap(vectors[("PCA_POSITIVE_ONLY_ROUTING",metric)],
                    vectors[("BASELINE",metric)],stable_seed("POSITIVE_VS_BASELINE",task,metric))
                contrasts.append({"task":task,"contrast":"POSITIVE_ONLY - BASELINE","comparator_variant":"BASELINE",
                    "metric":metric,"difference":delta,"CI95_low":lo,"CI95_high":hi,
                    "bootstrap_draws":BOOTSTRAPS,"unit":"biological subject","selected_best_random_draw":random_best,
                    "label":"POST_HELDOUT_DEVELOPMENT_EVALUATION"})
        # Record worst/mean/best random BA based on all five prespecified draws.
        for row in summary:
            if row["task"]==task and row["variant"]=="RANDOM_BASIS_0":
                row["random_draw_mean_BA"]=float(np.mean([vectors[(r,"BA")].mean() for r in random_names]))
                row["random_draw_best_variant"]=random_best
                row["random_draw_best_BA"]=float(vectors[(random_best,"BA")].mean())
                row["random_draw_worst_BA"]=float(vectors[(random_worst,"BA")].mean())
        for p in parts:p.close()
    cwrite(OUT/"HELDOUT_SUBJECT_RESULTS.csv",subjects_out)
    cwrite(OUT/"HELDOUT_MODEL_TASK_SUMMARY.csv",summary)
    cwrite(OUT/"PAIRED_HELDOUT_CONTRASTS.csv",contrasts)
    cwrite(OUT/"HELDOUT_RESCUE_HARM.csv",rescue)
    route_rows=list(csv.DictReader((OUT/"ROUTING_COEFFICIENTS.csv").open(newline="",encoding="utf-8")))
    basis_rows=list(csv.DictReader((OUT/"FINAL_REFIT_C_BASIS_AUDIT.csv").open(newline="",encoding="utf-8")))
    for row in movement:
        task=row["task"];fold=int(row["fold"]);rr=[r for r in route_rows if r["task"]==task and int(float(r["fold"]))==fold]
        plus=[r for r in rr if r["class"]=="C_PLUS"];minus=[r for r in rr if r["class"]=="C_MINUS"]
        zeros=[r for r in rr if r["class"]=="C_ZERO"]
        basisrec=next(r for r in basis_rows if r["task"]==task and int(r["fold"])==fold)
        v=np.asarray(json.loads(basisrec["explained_variance_fraction"]))
        row.update({"C_plus_count":len(plus),"C_minus_count":len(minus),"C_zero_count":len(zeros),
            "mean_r_C_plus":float(np.mean([float(r["primary_r"]) for r in plus])) if plus else 1.0,
            "mean_r_C_minus":float(np.mean([float(r["primary_r"]) for r in minus])) if minus else 1.0,
            "min_r":min(float(r["primary_r"]) for r in rr),"max_r":max(float(r["primary_r"]) for r in rr),
            "C_plus_variance_fraction":float(sum(float(r["variance_fraction"]) for r in plus)),
            "C_minus_variance_fraction":float(sum(float(r["variance_fraction"]) for r in minus),),
            "label":"POST_HOC_HELDOUT_DIAGNOSTIC"})
    cwrite(OUT/"ROUTING_MECHANISM_AUDIT.csv",movement)
    cwrite(OUT/"HELDOUT_SESSION_SUMMARY.csv",session_rows)
    make_report(lock,summary,contrasts)
    print("AGGREGATE_COMPLETE",len(subjects_out),len(summary),"scope=POST_HELDOUT_DEVELOPMENT_EVALUATION",flush=True)


def make_report(lock,summary,contrasts):
    table={(r["task"],r["variant"]):r for r in summary}
    contrast={(r["task"],r["contrast"],r["metric"]):r for r in contrasts}
    route_rows=list(csv.DictReader((OUT/"ROUTING_COEFFICIENTS.csv").open(newline="",encoding="utf-8")))
    basis_rows=list(csv.DictReader((OUT/"FINAL_REFIT_C_BASIS_AUDIT.csv").open(newline="",encoding="utf-8")))
    lines=["# Utility-Guided Selective Complement Routing (UGCR-V3)","",
        "**Status: POST_HELDOUT_DEVELOPMENT_EVALUATION.** The formal heldout cohort was accessed in prior V1/V2 work; this is not untouched confirmatory evidence. V3 architecture, utility signs, coefficients, shuffle/random controls, metrics, and evaluator were frozen in `V3_PROTOCOL_LOCK.json` before any V3 heldout EEG array was read.","",
        "EEGNet was the canonical V1/V2 final-refit checkpoint, fully frozen. No router, gate, adapter, classifier, backbone, or BN state was trained; V3 adds zero trainable parameters. Direction utility was re-estimated only on the non-final-heldout refit-training pool with the frozen V2.5 native-coordinate P-mediated margin estimand.","",
        "## Primary future-session subject-equal BA","",
        "| Task | Baseline | Positive-only | Shuffled | Random mean | UGCR-V3 | Δ vs Base (95% CI) | Δ vs Shuffled (95% CI) | Δ vs Random mean (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        b=table[task,"BASELINE"]["BA"];pos=table[task,"PCA_POSITIVE_ONLY_ROUTING"]["BA"];sh=table[task,"SHUFFLED_UTILITY_ROUTING"]["BA"]
        v3=table[task,"PCA_SIGNED_UTILITY_ROUTING"]["BA"]
        rand=table[task,"RANDOM_BASIS_MEAN"]["BA"]
        vals=[]
        for name in ("BASELINE","SHUFFLED","RANDOM_MEAN"):
            c=contrast[task,"UGCR_V3 - "+name,"BA"];vals.append(f"{c['difference']:+.4f} [{c['CI95_low']:+.4f}, {c['CI95_high']:+.4f}]")
        lines.append(f"| {task} | {b:.4f} | {pos:.4f} | {sh:.4f} | {rand:.4f} | {v3:.4f} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines += ["","## Final-refit routing composition","",
        "Counts and variance fractions below are averages across five folds; basis indices are fold-local.","",
        "| Task | C+ / fold | C- / fold | C0 / fold | mean r(C+) | mean r(C-) | C+ variance fraction | C- variance fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    bytask={t:[r for r in route_rows if r["task"]==t] for t in TASKS}
    for task in TASKS:
        rr=bytask[task]; bp=[r for r in rr if r["class"]=="C_PLUS"];bn=[r for r in rr if r["class"]=="C_MINUS"]
        k=sum(int(float(r["fold"])==0) for r in rr)
        plus=sum(1 for r in rr if r["class"]=="C_PLUS")/5;minus=sum(1 for r in rr if r["class"]=="C_MINUS")/5
        zero=k-plus-minus
        mp=float(np.mean([float(r["primary_r"]) for r in bp])) if bp else 1.0
        mn=float(np.mean([float(r["primary_r"]) for r in bn])) if bn else 1.0
        byfold={int(r["fold"]):r for r in basis_rows if r["task"]==task}
        pvar=[];nvar=[]
        for fold in FOLDS:
            vr=np.asarray(json.loads(byfold[fold]["explained_variance_fraction"]))
            foldrows=[r for r in rr if int(float(r["fold"]))==fold]
            pvar.append(sum(float(r["variance_fraction"]) for r in foldrows if r["class"]=="C_PLUS"))
            nvar.append(sum(float(r["variance_fraction"]) for r in foldrows if r["class"]=="C_MINUS"))
        lines.append(f"| {task} | {plus:.2f} | {minus:.2f} | {zero:.2f} | {mp:.4f} | {mn:.4f} | {np.mean(pvar):.4f} | {np.mean(nvar):.4f} |")
    lines += ["","## Paired biological-subject bootstrap contrasts","",
        "20,000 resamples within task; unit is the biological subject. Differences are UGCR-V3 minus comparator. Macro-F1, worst-session BA, and NLL are also in `PAIRED_HELDOUT_CONTRASTS.csv`.","",
        "| Task | Comparator | Δ BA [95% CI] | Δ Macro-F1 [95% CI] | Δ worst-session BA [95% CI] | Δ NLL |",
        "|---|---|---:|---:|---:|---:|"]
    for task in TASKS:
        for comp in ("BASELINE","POSITIVE_ONLY","SHUFFLED","RANDOM_MEAN","BEST_RANDOM"):
            b=contrast[task,"UGCR_V3 - "+comp,"BA"];f=contrast[task,"UGCR_V3 - "+comp,"macro_F1"]
            w=contrast[task,"UGCR_V3 - "+comp,"worst_session_BA"];n=contrast[task,"UGCR_V3 - "+comp,"NLL"]
            lines.append(f"| {task} | {comp} | {b['difference']:+.4f} [{b['CI95_low']:+.4f}, {b['CI95_high']:+.4f}] | {f['difference']:+.4f} [{f['CI95_low']:+.4f}, {f['CI95_high']:+.4f}] | {w['difference']:+.4f} [{w['CI95_low']:+.4f}, {w['CI95_high']:+.4f}] | {n['difference']:+.4f} |")
    states=[]
    for task in TASKS:
        base=contrast[task,"UGCR_V3 - BASELINE","BA"];sh=contrast[task,"UGCR_V3 - SHUFFLED","BA"]
        rm=contrast[task,"UGCR_V3 - RANDOM_MEAN","BA"];br=contrast[task,"UGCR_V3 - BEST_RANDOM","BA"]
        pos=contrast[task,"UGCR_V3 - POSITIVE_ONLY","BA"]
        if base["CI95_low"]>0 and sh["CI95_low"]>0 and rm["CI95_low"]>0 and br["CI95_low"]>0:
            state="SELECTIVE_ROUTING_ACTIONABLE"
        elif contrast[task,"POSITIVE_ONLY - BASELINE","BA"]["CI95_low"]>0 and pos["CI95_high"]<=0:
            state="POSITIVE_ROUTING_ONLY"
        elif sh["CI95_low"]<=0<=sh["CI95_high"] and rm["CI95_low"]<=0<=rm["CI95_high"]:
            state="UTILITY_NOT_DIRECTION_SPECIFIC"
        elif base["CI95_high"]<=0 and contrast[task,"POSITIVE_ONLY - BASELINE","BA"]["CI95_high"]<=0:
            state="UTILITY_STRUCTURE_NON_ACTIONABLE"
        else:
            state="TASK_DEPENDENT_SELECTIVE_ROUTING"
        states.append(state)
    overall=states[0] if len(set(states))==1 else "TASK_DEPENDENT_SELECTIVE_ROUTING"
    lines += ["","## Interpretation","",f"Overall state: **{overall}**. Per-task states: "+", ".join(f"{t}={s}" for t,s in zip(TASKS,states))+".",
        "The state rules were fixed in the source protocol before heldout access. The best random draw is selected post-hoc by task BA among the five prelocked draws and is descriptive. A functional margin utility is not evidence of biological causality.","",
        "## Heldout diagnostics","","`ROUTING_MECHANISM_AUDIT.csv` reports P movement and downstream C-preservation error, marked `POST_HOC_HELDOUT_DIAGNOSTIC`. `HELDOUT_RESCUE_HARM.csv` reports errors rescued and correct trials damaged relative to five-fold baseline probabilities, with C+/C-/high-utility activation energy on those trials. These are descriptive and did not change routing.","",
        "## Reproducibility","",f"Protocol lock SHA-256: `{sha(PROTOCOL/'V3_PROTOCOL_LOCK.json')}`. Source commit: `{lock['source_commit']}`. All nine variants use the same five-fold probability aggregation, and all baseline checkpoints match V1 and V2 byte-for-byte.",""]
    (OUT/"FINAL_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    # Efficiency is derived from the frozen parameter inventory and routed forward count.
    eff=[]
    for task in TASKS:
        for fold in FOLDS:
            d=output_cell(task,fold); rec=json.loads((d/"PROJECTOR_AUDIT.json").read_text())
            geom=np.load(d/"geometry.npz",allow_pickle=False)
            nparam=int(rec["baseline_state_parameter_count"])
            for v in VARIANTS:
                routed=0 if v=="BASELINE" else 1
                eff.append({"task":task,"fold":fold,"variant":v,"baseline_parameters":nparam,
                    "trainable_parameter_delta":0,"trainable_parameters":0,"frozen_parameters":nparam,
                    "F0_forwards_per_trial":1+routed,"extra_F0_forwards_per_trial":routed,
                    "routing_directions":int(geom["c_basis"].shape[1]),"model_training_steps":0,
                    "label":"POST_HELDOUT_DEVELOPMENT_EVALUATION"})
            geom.close()
    cwrite(OUT/"EFFICIENCY.csv",eff)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("preflight","prepare","lock","final-eval","aggregate"))
    parser.add_argument("--task",choices=TASKS);parser.add_argument("--fold",type=int,choices=FOLDS)
    args=parser.parse_args()
    if args.stage=="preflight": preflight(); return
    if args.stage=="lock": lock_protocol(); return
    if args.stage=="aggregate": aggregate(); return
    if args.task is None or args.fold is None: parser.error("prepare/final-eval require --task and --fold")
    if args.stage=="prepare": prepare(args.task,args.fold)
    else: final_eval(args.task,args.fold)


if __name__=="__main__": main()
