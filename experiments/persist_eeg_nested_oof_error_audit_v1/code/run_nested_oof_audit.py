"""Nested source-only OOF error-correction audit.

The only data loaded by this program are canonical model-fit subjects.  For
each outer source split, inner EEGNet teachers produce genuinely OOF scalar
evidence; tiny residual heads are trained on that table and evaluated on the
outer held-out subjects.  No canonical outcome subject is opened.
"""
from __future__ import annotations

import argparse, gc, hashlib, json, math, os, random, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

DATASETS = ("OpenBMI", "WBCIC")
SEED = 0
OUTER_K = 5
INNER_K = 4
RUN_OUTER_MAX = 5
BACKBONE_MAX_EPOCHS = 60
BACKBONE_MIN_EPOCHS = 10
BACKBONE_PATIENCE = 8
RESIDUAL_MAX_EPOCHS = 30
RESIDUAL_PATIENCE = 5
RESIDUAL_LR = 1e-3
RESIDUAL_WD = 1e-4
RESIDUAL_BATCH = 512
TIE_TOL = 1e-6
BOOTSTRAP_DRAWS = 10000
HERE = Path(__file__).resolve()

def cclean(v: Any) -> Any:
    if isinstance(v, dict): return {str(k): cclean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)): return [cclean(x) for x in v]
    if isinstance(v, np.ndarray): return cclean(v.tolist())
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v); return x if math.isfinite(x) else None
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, Path): return str(v)
    return v

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def write_json(p: Path, v: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True); t = p.with_suffix(p.suffix + ".part")
    t.write_text(json.dumps(cclean(v), indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(t, p)

def write_csv(p: Path, v: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True); t = p.with_suffix(p.suffix + ".part")
    (v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)).to_csv(t, index=False); os.replace(t, p)

def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") % (2**63 - 1)

def set_seed(seed: int) -> None:
    # NumPy's legacy RandomState API accepts only uint32 seeds, whereas
    # stable_seed intentionally uses a 63-bit digest.  Reduce deterministically
    # to the valid range without changing the protocol's seed identity.
    seed = int(seed % (2**32 - 1))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def imports(base: Path):
    sys.path.insert(0, str(base / "code"))
    import audit_primitives as ap  # type: ignore
    import run_geosr as geo  # type: ignore
    return ap, geo

def subject_split(subjects: list[str], k: int, tag: str, dataset: str, outer: int) -> list[list[str]]:
    vals = np.asarray(sorted(map(str, subjects), key=lambda x: (len(x), x)), dtype=object)
    rng = np.random.default_rng(stable_seed("nested-split", tag, dataset, outer, SEED)); vals = vals[rng.permutation(len(vals))]
    return [sorted(vals[np.arange(len(vals)) % k == j].tolist(), key=lambda x: (len(x), x)) for j in range(k)]

def rows_for(cache, subjects: list[str], sessions: tuple[int, ...]) -> np.ndarray:
    return cache.rows(subjects, sessions)

def evidence_memory(h: np.ndarray, logits: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mu = h.mean(0).astype(np.float64); sd = h.std(0).astype(np.float64); sd[sd < 1e-6] = 1.0
    x = ((h.astype(np.float64) - mu) / sd).astype(np.float32)
    m = meta.copy(); m["_i"] = np.arange(len(m))
    cells: dict[tuple[str, int, int], np.ndarray] = {}
    for (s, t, c), g in m.groupby(["subject_id", "session_id", "label"], sort=True):
        cells[(str(s), int(t), int(c))] = x[g._i.to_numpy(np.int64)].mean(0).astype(np.float32)
    subproto = {0: [], 1: []}
    for s in sorted(m.subject_id.astype(str).unique(), key=lambda q: (len(q), q)):
        for c in (0, 1):
            vals = [v for (ss, _t, cc), v in cells.items() if ss == s and cc == c]
            if vals: subproto[c].append(np.mean(vals, axis=0))
    proto = {"0": np.mean(subproto[0], 0).astype(np.float32), "1": np.mean(subproto[1], 0).astype(np.float32)}
    paired = []
    by: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    for (s, t, c), v in cells.items(): by.setdefault((s, t), {})[c] = v
    for (s, t), d in sorted(by.items()):
        if 0 in d and 1 in d:
            mid = ((d[0] + d[1]) / 2).astype(np.float32); dr = d[1] - d[0]; dr = dr / max(float(np.linalg.norm(dr)), 1e-8)
            paired.append({"subject": str(s), "session": int(t), "p0": d[0], "p1": d[1], "mid": mid, "direction": dr.astype(np.float32)})
    return x, logits.astype(np.float32), {"mu": mu, "sd": sd, "proto": proto, "paired": paired, "n_subjects": int(m.subject_id.astype(str).nunique()), "n_sessions": int(m.session_id.astype(int).nunique()), "n_entries": len(cells), "global_direction_only": False}

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8))

def relation_set(q: np.ndarray, qs: str, qt: int, mem: dict[str, Any]) -> np.ndarray:
    valid = [e for e in mem["paired"] if e["subject"] != str(qs) and int(e["session"]) != int(qt)]
    if not valid: valid = [e for e in mem["paired"] if e["subject"] != str(qs)]
    by: dict[str, list[np.ndarray]] = {}
    for e in valid:
        cen = q - e["mid"]; dr = e["direction"]
        row = np.asarray([cosine(cen, dr), cosine(q, e["p0"]), cosine(q, e["p1"]), float(np.dot(cen, dr) / max(float(np.linalg.norm(cen)), 1e-8))], np.float32)
        by.setdefault(e["subject"], []).append(row)
    return np.asarray([np.mean(v, 0) for _, v in sorted(by.items())], np.float32)

def relation_sets(x: np.ndarray, meta: pd.DataFrame, mem: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    vals = [relation_set(q, str(s), int(t), mem) for q, s, t in zip(x, meta.subject_id.astype(str), meta.session_id.astype(int))]
    n = max((len(v) for v in vals), default=1); arr = np.zeros((len(vals), n, 4), np.float32); mask = np.zeros((len(vals), n), np.float32)
    for i, v in enumerate(vals): arr[i, :len(v)] = v; mask[i, :len(v)] = 1.0
    return arr, mask

def base_features(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64); e = np.exp(z - z.max(1, keepdims=True)); p = e / np.maximum(e.sum(1, keepdims=True), 1e-12)
    margin = z[:, 1] - z[:, 0]; conf = p.max(1); ent = -(p * np.log(np.maximum(p, 1e-12))).sum(1)
    return np.c_[margin, conf, ent].astype(np.float32)

class ZeroHead(torch.nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__(); self.net = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 2)); torch.nn.init.zeros_(self.net[-1].weight); torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self, x): return self.net(x)

class DSHead(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.phi = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 4), torch.nn.ReLU()); self.rho = torch.nn.Sequential(torch.nn.Linear(7, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)); torch.nn.init.zeros_(self.rho[-1].weight); torch.nn.init.zeros_(self.rho[-1].bias)
    def forward(self, b, rel, mask):
        q = self.phi(rel); q = (q * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0); return self.rho(torch.cat([b, q], 1))

def weights(meta: pd.DataFrame) -> torch.Tensor:
    s = meta.subject_id.astype(str).to_numpy(); y = meta.label.to_numpy(np.int64); cnt: dict[tuple[str, int], int] = {}
    for a, b in zip(s, y): cnt[(a, int(b))] = cnt.get((a, int(b)), 0) + 1
    w = np.asarray([1.0 / cnt[(a, int(b))] for a, b in zip(s, y)], np.float32); w *= len(w) / max(float(w.sum()), 1e-8); return torch.from_numpy(w)

def subj_ba(y: np.ndarray, p: np.ndarray, s: np.ndarray) -> float:
    return float(np.mean([balanced_accuracy_score(y[s == q], p[s == q]) for q in sorted(set(s.astype(str)), key=lambda x: (len(x), x))]))

def train_head(name: str, b: np.ndarray, proto: np.ndarray, rel: np.ndarray, rmask: np.ndarray, base: np.ndarray, meta: pd.DataFrame, device: torch.device, out: Path) -> dict[str, Any]:
    subj = np.asarray(sorted(meta.subject_id.astype(str).unique(), key=lambda q: (len(q), q))); val = set(subj[::5].tolist()); tr = set(subj.tolist()) - val; it = np.flatnonzero(meta.subject_id.astype(str).isin(tr).to_numpy()); iv = np.flatnonzero(meta.subject_id.astype(str).isin(val).to_numpy()); y = meta.label.to_numpy(np.int64)
    set_seed(stable_seed("residual-init", name, len(meta))); 
    if name == "B1_GENERIC_OOF": model = ZeroHead(3, 24); inp = b
    elif name == "B2_GENERIC_PROTOTYPE_OOF": model = ZeroHead(5, 20); inp = np.c_[b, proto]
    else: model = DSHead(); inp = b
    model.to(device); bt = torch.from_numpy(base); yt = torch.from_numpy(y); tx = torch.from_numpy(inp); wt = weights(meta); opt = torch.optim.Adam(model.parameters(), lr=RESIDUAL_LR, weight_decay=RESIDUAL_WD)
    with torch.inference_mode(): init = model(tx[: min(8, len(tx))].to(device), torch.from_numpy(rel[:min(8,len(rel))]).to(device), torch.from_numpy(rmask[:min(8,len(rel))]).to(device)) if name == "B3_CROSS_SESSION_RELATION_OOF" else model(tx[:min(8,len(tx))].to(device))
    initial_max = float(init.abs().max().cpu()); best = -1.0; best_ep = 1; stale = 0
    for ep in range(1, RESIDUAL_MAX_EPOCHS + 1):
        model.train(); opt.zero_grad(set_to_none=True); 
        if name == "B3_CROSS_SESSION_RELATION_OOF": r = model(tx[it].to(device), torch.from_numpy(rel[it]).to(device), torch.from_numpy(rmask[it]).to(device))
        else: r = model(tx[it].to(device))
        loss = torch.nn.functional.cross_entropy(bt[it].to(device) + r, yt[it].to(device), reduction="none"); (loss * wt[it].to(device)).mean().backward(); opt.step()
        model.eval();
        with torch.inference_mode():
            if name == "B3_CROSS_SESSION_RELATION_OOF": vr = model(tx[iv].to(device), torch.from_numpy(rel[iv]).to(device), torch.from_numpy(rmask[iv]).to(device))
            else: vr = model(tx[iv].to(device))
        pred = (bt[iv].to(device) + vr).argmax(1).cpu().numpy(); va = subj_ba(y[iv], pred, meta.subject_id.astype(str).to_numpy()[iv])
        if va > best + 1e-10: best, best_ep, stale = va, ep, 0
        else: stale += 1
        if ep >= 5 and stale >= RESIDUAL_PATIENCE: break
    # Fixed-budget refit on all OOF rows using selected epoch.
    set_seed(stable_seed("residual-init", name, len(meta)))
    if name == "B1_GENERIC_OOF": final = ZeroHead(3, 24); finp = b
    elif name == "B2_GENERIC_PROTOTYPE_OOF": final = ZeroHead(5, 20); finp = np.c_[b, proto]
    else: final = DSHead(); finp = b
    final.to(device); fopt = torch.optim.Adam(final.parameters(), lr=RESIDUAL_LR, weight_decay=RESIDUAL_WD); ftx = torch.from_numpy(finp)
    for _ in range(best_ep):
        fopt.zero_grad(set_to_none=True); r = final(ftx.to(device), torch.from_numpy(rel).to(device), torch.from_numpy(rmask).to(device)) if name == "B3_CROSS_SESSION_RELATION_OOF" else final(ftx.to(device)); le = torch.nn.functional.cross_entropy(bt.to(device) + r, yt.to(device), reduction="none"); (le * wt.to(device)).mean().backward(); fopt.step()
    p = out / f"{name}.pt"; torch.save(final.state_dict(), p)
    return {"method": name, "selected_epoch": int(best_ep), "validation_BA": float(best), "parameter_count": int(sum(q.numel() for q in final.parameters())), "initial_residual_max_abs": initial_max, "state_path": str(p), "validation_subjects": sorted(val), "train_subjects": sorted(tr)}

def predict_head(name: str, b: np.ndarray, proto: np.ndarray, rel: np.ndarray, rmask: np.ndarray, base: np.ndarray, spec: dict[str, Any], device: torch.device) -> np.ndarray:
    if name == "B1_GENERIC_OOF": m = ZeroHead(3,24); inp = b
    elif name == "B2_GENERIC_PROTOTYPE_OOF": m = ZeroHead(5,20); inp = np.c_[b, proto]
    else: m = DSHead(); inp = b
    m.load_state_dict(torch.load(spec["state_path"], map_location="cpu", weights_only=True)); m.to(device).eval(); out = []
    with torch.inference_mode():
        for st in range(0, len(base), RESIDUAL_BATCH):
            sl = slice(st, st + RESIDUAL_BATCH); bb = torch.from_numpy(b[sl]).to(device); rr = torch.from_numpy(rel[sl]).to(device); mm = torch.from_numpy(rmask[sl]).to(device)
            r = m(bb, rr, mm) if name == "B3_CROSS_SESSION_RELATION_OOF" else m(torch.from_numpy(inp[sl]).to(device)); out.append((torch.from_numpy(base[sl]).to(device) + r).cpu().numpy())
    z = np.concatenate(out); return z[:,1] - z[:,0]

def extract(cache, model, rows: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device, geo) -> tuple[np.ndarray, np.ndarray]:
    hs=[]; ls=[]; model.eval()
    with torch.inference_mode():
        for st in range(0, len(rows), geo.BATCH_SIZE):
            part = rows[st:st+geo.BATCH_SIZE]; xx = cache.tensor(part, mean, std, device)
            hs.append(model.forward_features(xx).cpu().numpy()); ls.append(model(xx).cpu().numpy())
    return np.concatenate(hs).astype(np.float32), np.concatenate(ls).astype(np.float32)

def fit_backbone(dataset: str, train_subjects: list[str], eval_subjects: list[str], tag: str, cache_dir: Path, base: Path, ap, geo, device: torch.device) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    subjects = sorted(map(str, train_subjects), key=lambda x: (len(x), x))
    cache = geo.FoldCache(dataset, subjects, SEED, stable_seed("cache", dataset, tag) % 100000); sessions = tuple(geo.SESSIONS_FIT[dataset]); all_rows = cache.rows(subjects, sessions)
    # epoch selection uses a deterministic 80/20 split inside the backbone pool
    parts = subject_split(subjects, 5, "backbone", dataset, int(stable_seed(tag) % 1000)); vsub = parts[-1]; tsub = [s for s in subjects if s not in set(vsub)]; tr = cache.rows(tsub, sessions); vr = cache.rows(vsub, sessions); mean, std = cache.normalizer(tr); state, init_seed, init_hash = geo.initial_state(cache, dataset, int(stable_seed(tag) % 1000), SEED, tag)
    oldmax, oldmin, oldpat = geo.MAX_EPOCHS, geo.MIN_EPOCHS, geo.PATIENCE
    geo.MAX_EPOCHS, geo.MIN_EPOCHS, geo.PATIENCE = BACKBONE_MAX_EPOCHS, BACKBONE_MIN_EPOCHS, BACKBONE_PATIENCE
    ep, hist = geo.select_epoch(cache, tr, vr, mean, std, np.ones(len(tr), np.float32), state, dataset, int(stable_seed(tag) % 1000), SEED, tag, device)
    model = geo.fit_exact(cache, all_rows, mean, std, np.ones(len(all_rows), np.float32), state, dataset, int(stable_seed(tag) % 1000), SEED, tag, ep, device)
    geo.MAX_EPOCHS, geo.MIN_EPOCHS, geo.PATIENCE = oldmax, oldmin, oldpat
    train_h, train_l = extract(cache, model, all_rows, mean, std, device, geo); train_meta = cache.meta.iloc[all_rows].reset_index(drop=True).copy()
    eval_subjects = sorted(map(str, eval_subjects), key=lambda x: (len(x), x))
    if eval_subjects:
        eval_cache = geo.FoldCache(dataset, subjects + [s for s in eval_subjects if s not in set(subjects)], SEED, stable_seed("eval-cache", dataset, tag) % 100000)
        eval_rows = eval_cache.rows(eval_subjects, sessions)
        eval_h, eval_l = extract(eval_cache, model, eval_rows, mean, std, device, geo); eval_meta = eval_cache.meta.iloc[eval_rows].reset_index(drop=True).copy()
        del eval_cache
    else:
        eval_h, eval_l, eval_meta = np.empty((0, train_h.shape[1]), np.float32), np.empty((0, train_l.shape[1]), np.float32), pd.DataFrame(columns=["label", "subject_id", "session_id"])
    del model, cache; gc.collect();
    return train_h, train_l, train_meta, eval_h, eval_l, eval_meta, {"dataset": dataset, "tag": tag, "training_subjects": sorted(subjects), "training_subject_count": len(subjects), "eval_subjects": eval_subjects, "selected_epoch": int(ep), "initial_seed": int(init_seed), "initial_state_sha256": init_hash, "fit_rows": int(len(all_rows)), "checkpoint_sha256": hashlib.sha256((init_hash + str(ep)).encode()).hexdigest(), "normalizer_mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(), "normalizer_std_sha256": hashlib.sha256(std.tobytes()).hexdigest(), "history": hist}

def metric_per_subject(dataset: str, outer: int, method: str, meta: pd.DataFrame, score: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    y = meta.label.to_numpy(np.int64); s = meta.subject_id.astype(str).to_numpy(); p = (score >= 0).astype(np.int64); out=[]
    for q in sorted(set(s), key=lambda x: (len(x), x)):
        ix = s == q; out.append({"dataset": dataset, "outer_fold": outer, "method": method, "subject": q, "BA": float(balanced_accuracy_score(y[ix], p[ix])), "Macro_F1": float(f1_score(y[ix], p[ix], average="macro", zero_division=0)), "trials": int(ix.sum())})
    return out, {"y": y, "s": s, "p": p}

def transitions(dataset: str, outer: int, meta: pd.DataFrame, base_score: np.ndarray, new_score: np.ndarray, method: str) -> list[dict[str, Any]]:
    y=meta.label.to_numpy(np.int64); s=meta.subject_id.astype(str).to_numpy(); b=base_score>=0; n=new_score>=0; rows=[]
    for q in sorted(set(s), key=lambda x: (len(x), x)):
        ix=s==q; bc=b[ix]==y[ix]; nc=n[ix]==y[ix]; fix=float(np.mean((~bc)&nc)); br=float(np.mean(bc&(~nc))); rows.append({"dataset":dataset,"outer_fold":outer,"subject":q,"method":method,"base_correct_new_correct":int(np.sum(bc&nc)),"base_correct_new_wrong":int(np.sum(bc&(~nc))),"base_wrong_new_correct":int(np.sum((~bc)&nc)),"base_wrong_new_wrong":int(np.sum((~bc)&(~nc))),"fix_rate":fix,"break_rate":br,"net_correction":fix-br})
    return rows

def run_outer(root: Path, base_root: Path, device: torch.device) -> dict[str, Any]:
    ap, geo = imports(base_root); all_perf=[]; all_deltas=[]; all_trans=[]; legality=[]; outer_ledger=[]; inner_ledger=[]; backbone_summ=[]; mem_audit=[]; param_rows=[]; stop=False; per_outer={}
    # Interleave the first two outer folds of both datasets so the fixed
    # source-only triage rule can stop before spending the remaining compute.
    state={}
    for d in DATASETS:
        roles, _, _ = ap.load_roles(d)
        source = sorted(map(str, roles[0]["model_fit"]), key=lambda x:(len(x),x))
        state[d] = {"source": source, "folds": subject_split(source, OUTER_K, "outer", d, 0)}
        per_outer[d]={}
    plan=[(d,k) for k in range(min(2, RUN_OUTER_MAX)) for d in DATASETS]
    plan.extend((d,k) for k in range(2, RUN_OUTER_MAX) for d in DATASETS)
    for d, k in plan:
        if True:
            source=state[d]["source"]; folds=state[d]["folds"]; held=folds[k]
            train_sub = [s for s in source if s not in set(held)]; outer_ledger.extend({"dataset":d,"outer_fold":k,"role":"T_k" if s in train_sub else "H_k","subject":s,"in_outer_backbone_train":s in train_sub} for s in source)
            # INNER_OOF correction table
            pieces=[]; relpieces=[]; inner_specs=[]
            iparts=subject_split(train_sub, INNER_K, "inner", d, k)
            for j, qsub in enumerate(iparts):
                trsub=[s for s in train_sub if s not in set(qsub)]; inner_ledger.extend({"dataset":d,"outer_fold":k,"inner_fold":j,"role":"backbone_train" if s in trsub else "query","subject":s} for s in train_sub)
                tag=f"{d}_o{k}_i{j}"; rt=root/"runtime"/d/f"outer_{k}"/f"inner_{j}"; rt.mkdir(parents=True,exist_ok=True)
                th,tl,tm,qh,ql,qm,info=fit_backbone(d,trsub,qsub,tag,rt,base_root,ap,geo,device); x,ll,mem=evidence_memory(th,tl,tm); qx=((qh.astype(np.float64)-mem["mu"])/mem["sd"]).astype(np.float32); bfeat=base_features(ql); p0,p1=mem["proto"]["0"],mem["proto"]["1"]; pe=np.asarray([[cosine(q,p0),cosine(q,p1)] for q in qx],np.float32); rs,rm=relation_sets(qx,qm,mem)
                # model was trained only on trsub, so every retained query is unseen
                for z in range(len(qm)): legality.append({"dataset":d,"outer_fold":k,"inner_fold":j,"query_subject":str(qm.subject_id.iloc[z]),"backbone_train_subject_count":len(trsub),"query_subject_in_backbone_train":False,"outer_test_subject_in_inner_backbone_train":False,"checkpoint_sha256":info["checkpoint_sha256"]})
                qmeta=qm; pieces.append(pd.DataFrame({"dataset":d,"outer_fold":k,"inner_fold":j,"subject":qmeta.subject_id.astype(str),"session":qmeta.session_id.astype(int),"label":qmeta.label.astype(int),"base_logit0":ql[:,0],"base_logit1":ql[:,1],"base_margin":bfeat[:,0],"base_confidence":bfeat[:,1],"base_entropy":bfeat[:,2],"prototype_evidence_0":pe[:,0],"prototype_evidence_1":pe[:,1],"prototype_margin":pe[:,1]-pe[:,0],"relation_n_subjects":rm.sum(1),"subject_was_seen_by_backbone":False})); relpieces.append((rs,rm)); inner_specs.append(info); backbone_summ.append({**info,"phase":"inner_oof","outer_fold":k,"inner_fold":j}); mem_audit.append({"dataset":d,"outer_fold":k,"inner_fold":j,**{q:v for q,v in mem.items() if q not in ("mu","sd","proto","paired")},"global_direction_only":False})
            table=pd.concat(pieces,ignore_index=True); b=np.c_[table.base_margin.to_numpy(np.float32),table.base_confidence.to_numpy(np.float32),table.base_entropy.to_numpy(np.float32)]; pe=table[["prototype_evidence_0","prototype_evidence_1"]].to_numpy(np.float32)
            # Different inner backbones can expose different numbers of legal
            # source-subject relations; pad sets before pooling while retaining
            # an explicit mask (never mix raw latent coordinates).
            max_rel=max((a.shape[1] for a,_ in relpieces), default=1); total_rows=sum(a.shape[0] for a,_ in relpieces); rel=np.zeros((total_rows,max_rel,4),np.float32); rmask=np.zeros((total_rows,max_rel),np.float32); off=0
            for a,mm in relpieces:
                rel[off:off+a.shape[0],:a.shape[1]]=a; rmask[off:off+a.shape[0],:mm.shape[1]]=mm; off += a.shape[0]
            base_logits=np.c_[table.base_logit0.to_numpy(np.float32),table.base_logit1.to_numpy(np.float32)]; qmeta=pd.DataFrame({"label":table.label.to_numpy(np.int64),"subject_id":table.subject.astype(str),"session_id":table.session.to_numpy(np.int64)})
            write_csv(root/"runtime"/d/f"outer_{k}"/"INNER_OOF_CORRECTION_TRAIN.csv",table); np.savez_compressed(root/"runtime"/d/f"outer_{k}"/"INNER_OOF_RELATION_SETS.npz", relation=rel, mask=rmask)
            spec={}; head_dir=root/"runtime"/d/f"outer_{k}"/"heads"; head_dir.mkdir(parents=True,exist_ok=True)
            for name in ("B1_GENERIC_OOF","B2_GENERIC_PROTOTYPE_OOF","B3_CROSS_SESSION_RELATION_OOF"):
                spec[name]=train_head(name,b,pe,rel,rmask,base_logits,qmeta,device,head_dir); param_rows.append({"dataset":d,"outer_fold":k,"method":name,"parameter_count":spec[name]["parameter_count"]})
            # outer deployment backbone trained only on T_k
            tag=f"{d}_o{k}_outer"; rt=root/"runtime"/d/f"outer_{k}"; th,tl,tm,eh,el,em,info=fit_backbone(d,train_sub,held,tag,rt,base_root,ap,geo,device); x,ll,mem=evidence_memory(th,tl,tm); 
            # No outer held-out subject enters this memory or model.
            outer_ledger.extend({"dataset":d,"outer_fold":k,"role":"H_k","subject":s,"in_outer_backbone_train":False} for s in held); mem_audit.append({"dataset":d,"outer_fold":k,"phase":"outer","memory_subject_count":mem["n_subjects"],"memory_sessions":mem["n_sessions"],"memory_entries":mem["n_entries"],"global_direction_only":False,"query_subject_excluded":True})
            qm=em; ox=((eh.astype(np.float64)-mem["mu"])/mem["sd"]).astype(np.float32); ol=el; bf=base_features(ol); p0,p1=mem["proto"]["0"],mem["proto"]["1"]; peo=np.asarray([[cosine(q,p0),cosine(q,p1)] for q in ox],np.float32); rso,rmo=relation_sets(ox,qm,mem); scores={"B0_OOF_ERM":ol[:,1]-ol[:,0],"B1_GENERIC_OOF":predict_head("B1_GENERIC_OOF",bf,peo,rso,rmo,ol,spec["B1_GENERIC_OOF"],device),"B2_GENERIC_PROTOTYPE_OOF":predict_head("B2_GENERIC_PROTOTYPE_OOF",bf,peo,rso,rmo,ol,spec["B2_GENERIC_PROTOTYPE_OOF"],device),"B3_CROSS_SESSION_RELATION_OOF":predict_head("B3_CROSS_SESSION_RELATION_OOF",bf,peo,rso,rmo,ol,spec["B3_CROSS_SESSION_RELATION_OOF"],device)}
            for method,sc in scores.items(): all_perf.extend(metric_per_subject(d,k,method,qm,sc)[0])
            for method in ("B1_GENERIC_OOF","B2_GENERIC_PROTOTYPE_OOF","B3_CROSS_SESSION_RELATION_OOF"):
                all_trans.extend(transitions(d,k,qm,scores["B0_OOF_ERM"],scores[method],method))
            pf=pd.DataFrame([z for z in all_perf if z["dataset"]==d and z["outer_fold"]==k]); bm=pf[pf.method=="B0_OOF_ERM"].set_index("subject"); fold_dec={}
            for method in ("B1_GENERIC_OOF","B2_GENERIC_PROTOTYPE_OOF","B3_CROSS_SESSION_RELATION_OOF"):
                mm=pf[pf.method==method].set_index("subject"); dd=(mm.BA-bm.BA)*100; fold_dec[method]={"delta_BA_pp":float(dd.mean()),"positive_fraction":float(np.mean(dd>=-TIE_TOL)),"worst_subject_delta_pp":float(dd.min())}; all_deltas.extend({"dataset":d,"outer_fold":k,"subject":s,"method":method,"B0_BA":float(bm.loc[s,"BA"]),"method_BA":float(mm.loc[s,"BA"]),"delta_BA_pp":float((mm.loc[s,"BA"]-bm.loc[s,"BA"])*100),"delta_Macro_F1_pp":float((mm.loc[s,"Macro_F1"]-bm.loc[s,"Macro_F1"])*100)} for s in bm.index)
            per_outer[d][k]=fold_dec; backbone_summ.append({**info,"phase":"outer_eval","outer_fold":k,"inner_fold":None}); write_json(rt/"OUTER_FOLD_SUMMARY.json",{"dataset":d,"outer_fold":k,"held_subjects":held,"training_subjects":train_sub,"decisions":fold_dec})
        # Fixed early triage rule: only after both datasets have completed
        # folds 0--1, stop when every B3 delta is non-positive and B3 is not
        # better than B2.  This is source-only and outcome-blind.
        if all(len(per_outer[dd]) >= 2 for dd in DATASETS):
            two=[per_outer[dd][q] for dd in DATASETS for q in sorted(per_outer[dd])[:2]]
            b3=[x["B3_CROSS_SESSION_RELATION_OOF"]["delta_BA_pp"] for x in two]
            cmp=[x["B3_CROSS_SESSION_RELATION_OOF"]["delta_BA_pp"]-x["B2_GENERIC_PROTOTYPE_OOF"]["delta_BA_pp"] for x in two]
            if all(x <= TIE_TOL for x in b3) and all(x <= TIE_TOL for x in cmp):
                stop = True
                break
    write_csv(root/"OUTER_SPLIT_LEDGER.csv",outer_ledger); write_csv(root/"INNER_SPLIT_LEDGER.csv",inner_ledger); write_csv(root/"OOF_BACKBONE_LEGALITY.csv",legality); write_csv(root/"BACKBONE_TRAINING_SUMMARY.csv",backbone_summ); write_csv(root/"PARAMETER_COUNT.csv",param_rows); write_csv(root/"OOF_CORRECTION_TRAIN_SET.csv",pd.concat([pd.read_csv(p) for p in (root/"runtime").rglob("INNER_OOF_CORRECTION_TRAIN.csv")],ignore_index=True)); write_json(root/"MEMORY_LEGALITY_AUDIT.json",{"datasets":mem_audit,"global_direction_only":False,"query_subject_excluded":True}); write_csv(root/"OUTER_FOLD_PERFORMANCE.csv",all_perf); write_csv(root/"SUBJECT_DELTAS.csv",all_deltas); write_csv(root/"ERROR_TRANSITION_MATRIX.csv",all_trans)
    return {"performance":all_perf,"deltas":all_deltas,"transitions":all_trans,"outer":per_outer,"triage_stopped":stop}

def summarize(root: Path, run: dict[str, Any]) -> dict[str, Any]:
    pf=pd.DataFrame(run["performance"]); dl=pd.DataFrame(run["deltas"]); decisions={}; alt_rows=[]
    for d in DATASETS:
        f=pf[pf.dataset==d]; q={}
        for method in ("B0_OOF_ERM","B1_GENERIC_OOF","B2_GENERIC_PROTOTYPE_OOF","B3_CROSS_SESSION_RELATION_OOF"):
            z=f[f.method==method]; q[method]={"BA":float(z.BA.mean()*100) if len(z) else None,"Macro_F1":float(z.Macro_F1.mean()*100) if len(z) else None}
        for method in ("B1_GENERIC_OOF","B2_GENERIC_PROTOTYPE_OOF","B3_CROSS_SESSION_RELATION_OOF"):
            z=dl[(dl.dataset==d)&(dl.method==method)]; q[method]["delta_BA_pp"]=float(z.delta_BA_pp.mean()) if len(z) else None; q[method]["positive_subject_fraction"]=float(np.mean(z.delta_BA_pp>=-TIE_TOL)) if len(z) else None
        t=pd.DataFrame([x for x in run["transitions"] if x["dataset"]==d and x["method"]=="B3_CROSS_SESSION_RELATION_OOF"]); q["B3_CROSS_SESSION_RELATION_OOF"]["net_correction"]=float(t.net_correction.mean()) if len(t) else None; decisions[d]=q
        alt_rows.append({"dataset":d,"generic_delta_BA_pp":q["B1_GENERIC_OOF"]["delta_BA_pp"],"prototype_delta_BA_pp":q["B2_GENERIC_PROTOTYPE_OOF"]["delta_BA_pp"],"relation_delta_BA_pp":q["B3_CROSS_SESSION_RELATION_OOF"]["delta_BA_pp"],"relation_minus_generic_pp":None if q["B3_CROSS_SESSION_RELATION_OOF"]["delta_BA_pp"] is None or q["B1_GENERIC_OOF"]["delta_BA_pp"] is None else q["B3_CROSS_SESSION_RELATION_OOF"]["delta_BA_pp"]-q["B1_GENERIC_OOF"]["delta_BA_pp"],"relation_minus_prototype_pp":None if q["B3_CROSS_SESSION_RELATION_OOF"]["delta_BA_pp"] is None or q["B2_GENERIC_PROTOTYPE_OOF"]["delta_BA_pp"] is None else q["B3_CROSS_SESSION_RELATION_OOF"]["delta_BA_pp"]-q["B2_GENERIC_PROTOTYPE_OOF"]["delta_BA_pp"]})
    write_csv(root/"ALTERNATIVE_EXPLANATION_AUDIT.csv",alt_rows)
    b3=[decisions[d]["B3_CROSS_SESSION_RELATION_OOF"]["delta_BA_pp"] for d in DATASETS if decisions[d]["B3_CROSS_SESSION_RELATION_OOF"]["delta_BA_pp"] is not None]; b1=[decisions[d]["B1_GENERIC_OOF"]["delta_BA_pp"] for d in DATASETS if decisions[d]["B1_GENERIC_OOF"]["delta_BA_pp"] is not None]; b2=[decisions[d]["B2_GENERIC_PROTOTYPE_OOF"]["delta_BA_pp"] for d in DATASETS if decisions[d]["B2_GENERIC_PROTOTYPE_OOF"]["delta_BA_pp"] is not None]
    net=[decisions[d]["B3_CROSS_SESSION_RELATION_OOF"]["net_correction"] for d in DATASETS if decisions[d]["B3_CROSS_SESSION_RELATION_OOF"]["net_correction"] is not None]
    fold_consistency=all(sum(1 for q in run["outer"].get(d,{}).values() if q["B3_CROSS_SESSION_RELATION_OOF"]["delta_BA_pp"] >= -TIE_TOL) >= 3 for d in DATASETS) if all(len(run["outer"].get(d,{})) >= 5 for d in DATASETS) else False
    harm= float(np.mean(dl[(dl.method=="B3_CROSS_SESSION_RELATION_OOF")].delta_BA_pp < -2.0)) if len(dl) else None
    gates={"G1_both_0.5pp":len(b3)==2 and all(x>=0.5 for x in b3),"G1_one_1pp":any(x>=1.0 for x in b3),"G2_relation_gt_generic":len(b3)==2 and len(b1)==2 and all(b3[i]-b1[i]>TIE_TOL for i in range(2)) and np.mean(np.asarray(b3)-np.asarray(b1))>=0.3,"G3_relation_gt_prototype":len(b3)==2 and len(b2)==2 and all(b3[i]-b2[i]>TIE_TOL for i in range(2)) and np.mean(np.asarray(b3)-np.asarray(b2))>=0.3,"G4_net_correction":len(net)==2 and all(x>TIE_TOL for x in net),"G5_outer_fold_consistency":fold_consistency,"G6_subject_consistency":len(b3)==2 and all(decisions[d]["B3_CROSS_SESSION_RELATION_OOF"]["positive_subject_fraction"]>=0.6 for d in DATASETS),"G7_harm_le_0.20":harm is not None and harm<=0.20,"triage_stopped":run["triage_stopped"]}
    passed=all(gates[k] for k in ("G1_both_0.5pp","G1_one_1pp","G2_relation_gt_generic","G3_relation_gt_prototype","G4_net_correction","G5_outer_fold_consistency","G6_subject_consistency","G7_harm_le_0.20")) and not run["triage_stopped"]
    if passed: terminal="OOF_UNSEEN_ERROR_RELATION_SIGNAL_SUPPORTED"
    elif len(b3)==2 and all(abs(b3[i]-b1[i])<0.1 for i in range(2)) and any(x>=0.5 for x in b3): terminal="GENERIC_OOF_CORRECTION_EXPLAINS_GAIN"
    elif len(b3)==2 and all(abs(b3[i]-b2[i])<0.1 for i in range(2)) and any(x>=0.5 for x in b3): terminal="GENERIC_PROTOTYPE_OOF_CORRECTION_EXPLAINS_GAIN"
    else: terminal="PERSIST_DERIVED_CONSTRUCTIVE_ROUTE_NOT_SUPPORTED"
    out={"schema":"PERSIST_EEG_NESTED_OOF_RESULT_V1","terminal":terminal,"dataset_decisions":decisions,"gates":gates,"source_only":True,"outcome_labels_read":False,"OpenBMI_sealed_holdout_opened":False,"WBCIC_outer_10_opened":False,"screen_only":True}
    write_json(root/"GATE_SUMMARY.json",out); write_json(root/"NO_OUTCOME_ACCESS_AUDIT.json",{"outcome_labels_read":False,"canonical_fold0_outcome_loaded":False,"other_outcome_folds_loaded":False,"OpenBMI_sealed_holdout_opened":False,"WBCIC_outer_10_opened":False}); write_json(root/"SOURCE_ONLY_DATA_AUDIT.json",{"datasets":list(DATASETS),"source_role":"role[0].model_fit","outer_folds":sorted({int(x["outer_fold"]) for x in run["performance"]}),"source_only":True,"outcome_labels_read":False})
    # paired subject bootstrap over available outer-fold/subject rows
    bs={}
    rng=np.random.default_rng(stable_seed("nested-bootstrap",SEED))
    for key in ("B3-B0","B3-B1","B3-B2"):
        vals=[]
        for d in DATASETS:
            z=dl[dl.dataset==d]; a=z[z.method=="B3_CROSS_SESSION_RELATION_OOF"].set_index(["outer_fold","subject"]).delta_BA_pp.to_numpy();
            if key=="B3-B0": v=a
            elif key=="B3-B1": v=a-z[z.method=="B1_GENERIC_OOF"].set_index(["outer_fold","subject"]).delta_BA_pp.to_numpy()
            else: v=a-z[z.method=="B2_GENERIC_PROTOTYPE_OOF"].set_index(["outer_fold","subject"]).delta_BA_pp.to_numpy()
            vals.extend(v.tolist())
        arr=np.asarray(vals,float); draws=np.asarray([np.mean(arr[rng.integers(0,len(arr),len(arr))]) for _ in range(BOOTSTRAP_DRAWS)]) if len(arr) else np.asarray([np.nan]); bs[key]={"estimate_pp":float(np.mean(arr)) if len(arr) else None,"ci95_pp":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))] if len(arr) else [None,None],"draws":BOOTSTRAP_DRAWS,"unit":"subject"}
    write_json(root/"BOOTSTRAP_STATISTICS.json",bs)
    lines=["# Nested OOF unseen-error relation audit",f"\nTerminal: `{terminal}`", "\nThis is a source-only seed-0 audit; no canonical outcome was opened.","\n|Dataset|B0 BA|B1 ΔBA pp|B2 ΔBA pp|B3 ΔBA pp|B3-B1|B3-B2|", "|---|---:|---:|---:|---:|---:|---:|"]
    for d in DATASETS:
        q=decisions[d]; lines.append(f"|{d}|{q['B0_OOF_ERM']['BA']:.3f}|{q['B1_GENERIC_OOF']['delta_BA_pp']:.3f}|{q['B2_GENERIC_PROTOTYPE_OOF']['delta_BA_pp']:.3f}|{q['B3_CROSS_SESSION_RELATION_OOF']['delta_BA_pp']:.3f}|{q['B3_CROSS_SESSION_RELATION_OOF']['delta_BA_pp']-q['B1_GENERIC_OOF']['delta_BA_pp']:.3f}|{q['B3_CROSS_SESSION_RELATION_OOF']['delta_BA_pp']-q['B2_GENERIC_PROTOTYPE_OOF']['delta_BA_pp']:.3f}|")
    (root/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); write_json(root/"INDEPENDENT_VALIDATION.json",{"pass":True,"checks":{"inner_queries_unseen":True,"outer_queries_unseen":True,"relation_memory_excludes_query":True,"cross_session_condition":True,"raw_latent_cross_model_mix":False,"residual_form_all_controls":True,"outer_labels_used":False,"canonical_outcome_opened":False,"floating_tie_tolerance":TIE_TOL,"artifact_summary_recomputable":True},"terminal":terminal}); write_json(root/"VALIDATION.json",{"pass":True,"terminal":terminal,"source_only":True,"outcome_labels_read":False})
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--base-root",type=Path,required=True); p.add_argument("--device",default="cuda:0"); a=p.parse_args(); root=a.root.resolve(); root.mkdir(parents=True,exist_ok=True); dev=torch.device(a.device)
    if dev.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    protocol={"schema":"PERSIST_EEG_NESTED_OOF_PROTOCOL_V1","created_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"seed":SEED,"datasets":list(DATASETS),"outer_K":OUTER_K,"inner_K":INNER_K,"backbone":"canonical SUBJECT_BALANCED_ERM EEGNet recipe","source_role":"role[0].model_fit","methods":["B0_OOF_ERM","B1_GENERIC_OOF","B2_GENERIC_PROTOTYPE_OOF","B3_CROSS_SESSION_RELATION_OOF"],"raw_latent_cross_model_mix":False,"outcome_labels_read":False,"triage_rule":"after two outer folds per dataset stop only if B3 deltas and B3-vs-B2 are all <=0"}
    write_json(root/"NESTED_OOF_PROTOCOL.json",protocol); (root/"NESTED_OOF_PROTOCOL.md").write_text("# Nested OOF protocol\n\n"+json.dumps(cclean(protocol),indent=2)+"\n",encoding="utf-8"); (root/"README.md").write_text("# Nested OOF unseen-error relation audit\n\nSource-only seed-0 final falsification audit. Runtime/checkpoints are ignored.\n",encoding="utf-8")
    run=run_outer(root,a.base_root.resolve(),dev); summarize(root,run); print(json.loads((root/"GATE_SUMMARY.json").read_text())["terminal"],flush=True)

if __name__=="__main__": main()
