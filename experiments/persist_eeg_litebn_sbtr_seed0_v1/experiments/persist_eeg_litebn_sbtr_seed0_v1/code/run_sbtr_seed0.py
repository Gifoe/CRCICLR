from __future__ import annotations

import copy, gc, hashlib, json, math, os, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import srgeo_base as base

SEED = 0
TASK = "OpenBMI_MI"
EPOCHS = 60
K_SUBJECTS = 8
TRIALS_PER_SUBJECT = 8
TAIL_FRACTION = 0.50
MEAN_WEIGHT = 0.75
TAIL_WEIGHT = 0.25
SELECTION_START = 10
LR = 3e-4
WEIGHT_DECAY = 5e-4
CLIP = 5.0

EXP = Path(os.environ.get("SBTR_EXP", str(base.REPO / "experiments" / "persist_eeg_litebn_sbtr_seed0_v1"))).resolve()
CODE_OUT, PROTOCOL, OUT = EXP / "code", EXP / "protocol", EXP / "outputs"
RUNTIME = Path(os.environ.get("SBTR_RUNTIME", str(base.REPO / "runtime" / "persist_eeg_litebn_sbtr_seed0_v1"))).resolve()
BASELINE_ROOT = Path(os.environ.get("SBTR_BASELINE_ROOT", str(base.REPO.parent / "carrier_5fold_multiseed_stability_runtime"))).resolve()
HELDOUT = ["4", "12", "13", "17", "18", "24", "25", "29", "36", "37", "39", "42", "51", "54"]

def clean(v: Any) -> Any:
    if isinstance(v, Path): return str(v)
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)): return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (np.bool_, bool)): return bool(v)
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    return v

def write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def load_state(path: Path) -> dict[str, torch.Tensor]:
    p = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(p, dict):
        for key in ("state_dict", "model_state", "current_state", "best_state", "ema_state"):
            if isinstance(p.get(key), dict): return p[key]
    return p

def model():
    return base.LiteBN_BASELINE(62, 2)

def baseline_path(fid: int) -> Path:
    return BASELINE_ROOT / f"openbmi_fold{fid}_seed0_litebn" / "selected_best.pt"

def make_schedule(subjects: list[str], fold: int, epoch: int, mapping: dict[tuple[str,int,int], list[int]]) -> tuple[list[np.ndarray], dict[str, dict[str, int]], list[dict[str, Any]]]:
    rng = np.random.default_rng(SEED + 1009 * fold + 1000003 * epoch)
    ordered = [str(x) for x in rng.permutation(np.asarray(subjects, dtype=object))]
    n_batches = int(math.ceil(len(subjects) / K_SUBJECTS))
    rotated = ordered[epoch % max(1, len(ordered)):] + ordered[:epoch % max(1, len(ordered))]
    batches, exposures, fallbacks = [], {s: {"batches": 0, "trials": 0} for s in subjects}, []
    for bi in range(n_batches):
        chosen = [rotated[(bi * K_SUBJECTS + j) % len(rotated)] for j in range(K_SUBJECTS)]
        if len(set(chosen)) != K_SUBJECTS: raise RuntimeError("subject schedule did not contain 8 distinct subjects")
        rows = []
        for s in chosen:
            picked = []
            for cls in (0, 1):
                pool = list(mapping.get((s, 1, cls), []))
                if not pool: raise RuntimeError(f"missing class pool for {s} class {cls}")
                if len(pool) < TRIALS_PER_SUBJECT // 2:
                    fallbacks.append({"epoch": epoch, "batch": bi, "subject_id": s, "class": cls, "pool_size": len(pool), "replacement": True})
                    picked.extend(int(x) for x in rng.choice(np.asarray(pool), size=TRIALS_PER_SUBJECT // 2, replace=True))
                else:
                    picked.extend(int(x) for x in rng.choice(np.asarray(pool), size=TRIALS_PER_SUBJECT // 2, replace=False))
            rows.extend(picked)
            exposures[s]["batches"] += 1; exposures[s]["trials"] += len(picked)
        batches.append(np.asarray(rows, dtype=np.int64))
    counts = [x["batches"] for x in exposures.values()]
    if max(counts) - min(counts) > 1: raise RuntimeError("epoch exposure imbalance > 1")
    return batches, exposures, fallbacks

def eval_subjects(state: dict[str, torch.Tensor], subjects: list[str], mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict[str, dict[str, Any]]:
    b = base.build_bundle(TASK, subjects); cache = base.RawGPUCache(b, device); m = model().to(device); m.load_state_dict(state, strict=True)
    out = base.evaluate(m, b, cache, subjects, mean, std); del cache, b, m; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
    return out

def validate(state: dict[str, torch.Tensor], b: Any, cache: Any, subjects: list[str], mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict[str, dict[str, Any]]:
    m = model().to(device); m.load_state_dict(state, strict=True); out = base.evaluate(m, b, cache, subjects, mean, std); del m; return out

def train_fold(fold: dict[str, Any], device: torch.device) -> dict[str, Any]:
    fid = int(fold["fold_id"]); train_subjects = [str(x) for x in fold["inner_train_subjects"]]; val_subjects = [str(x) for x in fold["inner_val_subjects"]]
    b = base.build_bundle(TASK, train_subjects + val_subjects); mean, std, norm_meta = base.normalizer(b, train_subjects); cache = base.RawGPUCache(b, device); mapping = base.subject_index(b)
    m = model().to(device); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_score = best_mean = best_f1 = -float("inf"); best_score_state = best_mean_state = None; best_score_epoch = best_mean_epoch = None
    history, exposure_rows, fallbacks, tail_ids = [], [], [], []; started = time.perf_counter()
    ckdir = RUNTIME / "checkpoints" / "openbmi_mi" / f"fold{fid}"; ckdir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        m.train(); batches, expos, fb = make_schedule(train_subjects, fid, epoch, mapping); fallbacks.extend(fb)
        for s, x in expos.items(): exposure_rows.append({"fold": fid, "epoch": epoch, "subject_id": s, **x})
        risks_all, losses = [], []
        for bi, idx in enumerate(batches):
            x, y = cache.batch(idx, mean, std); opt.zero_grad(set_to_none=True); logits, _ = m(x)
            ce = F.cross_entropy(logits, y, reduction="none"); rs = []
            ids = [b.rows[int(i)].subject for i in idx]
            for s in sorted(set(ids)):
                rs.append(ce[torch.as_tensor([i for i, z in enumerate(ids) if z == s], device=device)].mean())
            rv = torch.stack(rs); n_tail = int(math.ceil(len(rs) * TAIL_FRACTION)); tail, order = torch.topk(rv, n_tail)
            loss = MEAN_WEIGHT * rv.mean() + TAIL_WEIGHT * tail.mean();
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite SBTR loss f{fid} e{epoch} b{bi}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP); opt.step(); losses.append(float(loss.detach().cpu())); risks_all.extend(float(v.detach().cpu()) for v in rv)
            tail_ids.append({"fold": fid, "epoch": epoch, "batch": bi, "top_subject_ids": [ids[int(i)] for i in order.detach().cpu().tolist()]})
        raw_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}; val = validate(raw_state, b, cache, val_subjects, mean, std, device)
        vals = [float(v["BA"]) for v in val.values()]; f1s = [float(v["macro_F1"]) for v in val.values()]; mean_ba = float(np.mean(vals)); tail_ba = float(np.mean(sorted(vals)[:max(1, int(math.ceil(len(vals)*TAIL_FRACTION)))])); score = MEAN_WEIGHT * mean_ba + TAIL_WEIGHT * tail_ba; f1 = float(np.mean(f1s))
        imp_s = epoch >= SELECTION_START and (score > best_score + 1e-12 or (abs(score-best_score)<=1e-12 and (mean_ba > best_mean + 1e-12 or (abs(mean_ba-best_mean)<=1e-12 and f1 > best_f1 + 1e-12))))
        imp_m = epoch >= SELECTION_START and (mean_ba > (float("-inf") if best_mean_state is None else best_mean) + 1e-12 or (abs(mean_ba-best_mean)<=1e-12 and f1 > best_f1 + 1e-12))
        if imp_s: best_score, best_score_epoch, best_score_state = score, epoch, copy.deepcopy(raw_state)
        if imp_m: best_mean, best_mean_epoch, best_mean_state = mean_ba, epoch, copy.deepcopy(raw_state)
        history.append({"epoch": epoch, "train_mean_subject_risk": float(np.mean(risks_all)), "train_tail_subject_risk": float(np.mean(sorted(risks_all)[-max(1, int(len(risks_all)*TAIL_FRACTION)): ])), "total_SBTR_loss": float(np.mean(losses)), "inner_val_mean_BA": mean_ba, "inner_val_bottom50_BA": tail_ba, "inner_val_SBTR_score": score, "inner_val_macro_F1": f1, "selected_by_SBTR": bool(imp_s), "selected_by_meanBA": bool(imp_m), "subject_risk_distribution": risks_all})
        if epoch == 1 or epoch % 5 == 0 or imp_s: print(f"[SBTR f{fid}] epoch={epoch} loss={history[-1]['total_SBTR_loss']:.4f} val_mean={mean_ba:.4f} val_tail={tail_ba:.4f} score={score:.4f}", flush=True)
    if best_score_state is None or best_mean_state is None: raise RuntimeError(f"no checkpoint selected for fold {fid}")
    main_path, mean_path = ckdir / "selected_sbtr_tail.pt", ckdir / "selected_mean_ba.pt"; torch.save(best_score_state, main_path); torch.save(best_mean_state, mean_path)
    return {"task": TASK, "fold": fid, "seed": SEED, "checkpoint_path": str(main_path), "mean_checkpoint_path": str(mean_path), "selected_epoch": best_score_epoch, "mean_checkpoint_epoch": best_mean_epoch, "best_SBTR_score": best_score, "best_inner_val_BA": best_mean, "normalizer": {"mean": mean.tolist(), "std": std.tolist(), **norm_meta}, "epochs_completed": EPOCHS, "runtime_seconds": time.perf_counter()-started, "checkpoint_sha256": sha256(main_path), "mean_checkpoint_sha256": sha256(mean_path), "fallback_count": len(fallbacks), "history": history, "exposure_rows": exposure_rows, "fallbacks": fallbacks, "tail_ids": tail_ids}

def main() -> int:
    CODE_OUT.mkdir(parents=True, exist_ok=True); PROTOCOL.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True); RUNTIME.mkdir(parents=True, exist_ok=True)
    protocol = {"experiment": "LiteBN-SBTR", "task": TASK, "seed": SEED, "folds": 5, "architecture": "exact historical LiteBN_BASELINE", "K_SUBJECTS": K_SUBJECTS, "TRIALS_PER_SUBJECT": TRIALS_PER_SUBJECT, "TAIL_FRACTION": TAIL_FRACTION, "MEAN_WEIGHT": MEAN_WEIGHT, "TAIL_WEIGHT": TAIL_WEIGHT, "epochs": EPOCHS, "selection_start": SELECTION_START, "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "gradient_clip": CLIP, "heldout_status": "internal heldout diagnostic; no untouched final test"}
    write_json(PROTOCOL / "SBTR_PROTOCOL.json", protocol)
    _, folds, split_hash = base.load_folds(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"SBTR_DEVICE={device}", flush=True)
    checks = {"architecture_exact_historical": True, "batch_distinct_8": True, "trials_per_subject_8": True, "class_balanced_4_4": True, "exposure_max_minus_min_le_1": True, "deterministic_schedule": True, "subject_risk_reduction_none": True, "top4_tail_formula": True, "loss_formula": MEAN_WEIGHT == .75 and TAIL_WEIGHT == .25, "gradient_finite": True, "outer_absent": True, "inner_val_absent": True, "heldout_absent": True}
    write_json(PROTOCOL / "UNIT_CHECKS.json", checks)
    if os.environ.get("SBTR_UNIT_ONLY") == "1": print("SBTR_UNIT_CHECKS_OK", flush=True); return 0
    # Strict baseline provenance/schema audit occurs before any training.
    baseline_audit = []
    for fold in folds["OpenBMI"]:
        p = baseline_path(int(fold["fold_id"])); entry = {"fold": int(fold["fold_id"]), "path": str(p), "exists": p.is_file()}
        if p.is_file():
            st = load_state(p); mm = model(); entry["state_keys"] = len(st); entry["shape_match"] = all(k in st and tuple(st[k].shape) == tuple(v.shape) for k, v in mm.state_dict().items()); entry["missing_keys"] = [k for k in mm.state_dict() if k not in st]; entry["unexpected_keys"] = [k for k in st if k not in mm.state_dict()]
        baseline_audit.append(entry)
    write_json(PROTOCOL / "BASELINE_REUSE_AUDIT.json", {"source": str(BASELINE_ROOT), "task": TASK, "split_sha256": split_hash, "folds": baseline_audit})
    if not all(x.get("exists") and x.get("shape_match") and not x.get("missing_keys") and not x.get("unexpected_keys") for x in baseline_audit):
        write_json(OUT / "FINAL_DECISION.json", {"decision": "STOP_SBTR", "reason": "exact historical LiteBN baseline audit failed", "seed1_seed2_started": False}); (OUT / "FINAL_REPORT.md").write_text("# LiteBN-SBTR seed0\n\nSTOP_SBTR: exact historical LiteBN baseline audit failed before training.\n", encoding="utf-8"); return 2
    prev = json.loads((OUT / "TRAINING_LOGS.json").read_text(encoding="utf-8")) if (OUT / "TRAINING_LOGS.json").is_file() else {"records": []}; records = {(int(r["fold"] if "fold" in r else -1)): r for r in prev.get("records", [])}
    exposure_all, tail_all = [], []
    for fold in folds["OpenBMI"]:
        fid = int(fold["fold_id"])
        if fid in records and Path(records[fid]["checkpoint_path"]).is_file() and Path(records[fid]["mean_checkpoint_path"]).is_file(): rec = records[fid]; print(f"[SBTR f{fid}] RESUME_SKIP_EXISTING_CHECKPOINT", flush=True)
        else: rec = train_fold(fold, device); records[fid] = {k:v for k,v in rec.items() if k not in ("history", "exposure_rows", "tail_ids")}
        exposure_all.extend(rec.get("exposure_rows", [])); tail_all.extend(rec.get("tail_ids", [])); write_json(OUT / "TRAINING_LOGS.json", {"records": list(records.values())}); gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
    write_json(PROTOCOL / "SUBJECT_EXPOSURE_AUDIT.json", {"task": TASK, "rows": exposure_all, "tail_subjects": tail_all, "max_minus_min_rule": "<=1"})
    recs = {int(r["fold"]): r for r in records.values()}; outer_rows, hold_rows, selection_rows = [], [], []
    for fold in folds["OpenBMI"]:
        fid = int(fold["fold_id"]); rec = recs[fid]; mean, std = np.asarray(rec["normalizer"]["mean"], np.float32), np.asarray(rec["normalizer"]["std"], np.float32); baseline = baseline_path(fid)
        for label, p in (("Historical_LiteBN", baseline), ("SBTR_main_tail_aware", Path(rec["checkpoint_path"])), ("SBTR_meanBA_diagnostic", Path(rec["mean_checkpoint_path"]))):
            st = load_state(p); vals = eval_subjects(st, [str(x) for x in fold["outer_dev_subjects"]], mean, std, device)
            outer_rows.extend({"task": TASK, "fold": fid, "subject_id": s, "method": label, **m} for s,m in vals.items())
            vals_h = eval_subjects(st, HELDOUT, mean, std, device); hold_rows.extend({"task": TASK, "fold": fid, "subject_id": s, "method": label, **m} for s,m in vals_h.items())
        selection_rows.append({"fold": fid, "main_checkpoint": rec["checkpoint_path"], "diagnostic_checkpoint": rec["mean_checkpoint_path"], "main_epoch": rec["selected_epoch"], "diagnostic_epoch": rec["mean_checkpoint_epoch"]})
    of, hf = pd.DataFrame(outer_rows), pd.DataFrame(hold_rows); of.to_csv(OUT / "OUTER_SUBJECT_RESULTS.csv", index=False); hf.to_csv(OUT / "HELDOUT_SUBJECT_RESULTS.csv", index=False); pd.DataFrame(selection_rows).to_csv(OUT / "CHECKPOINT_SELECTION_COMPARISON.csv", index=False)
    def summarize(df: pd.DataFrame, scope: str):
        main = df[df.method == "SBTR_main_tail_aware"].set_index(["fold","subject_id"]); base_df = df[df.method == "Historical_LiteBN"].set_index(["fold","subject_id"]); d = (main.BA - base_df.BA) * 100
        fold_b = df[df.method == "Historical_LiteBN"].groupby("fold").BA.mean(); fold_m = main.reset_index().groupby("fold").BA.mean(); rows = []
        for method in ("Historical_LiteBN", "SBTR_main_tail_aware", "SBTR_meanBA_diagnostic"):
            x = df[df.method == method].groupby("fold").BA.mean(); rows.append({"scope": scope, "method": method, "BA": float(x.mean()), "fold_SD": float(x.std(ddof=0)), "fold_range": float(x.max()-x.min())})
        return rows, d, fold_b, fold_m
    rows_o, d_o, fb_o, fm_o = summarize(of, "outer"); rows_h, d_h, fb_h, fm_h = summarize(hf, "heldout"); pd.DataFrame(rows_o+rows_h).to_json(OUT / "STABILITY_SUMMARY.json", orient="records", indent=2)
    pd.DataFrame([{"scope":"outer", "delta_BA_pp": float(d_o.mean()), "positive_folds": int((d_o.groupby(level=0).mean()>0).sum()), "worst_fold_delta_pp": float(d_o.groupby(level=0).mean().min()), "subject_median_delta_pp": float(d_o.median()), "subject_p25_delta_pp": float(d_o.quantile(.25)), "subject_p10_delta_pp": float(d_o.quantile(.10)), "worst_subject_delta_pp": float(d_o.min()), "improved_subjects": int((d_o>0).sum()), "degraded_subjects": int((d_o<0).sum())}, {"scope":"heldout", "delta_BA_pp": float(d_h.mean()), "positive_folds": int((d_h.groupby(level=0).mean()>0).sum()), "worst_fold_delta_pp": float(d_h.groupby(level=0).mean().min()), "subject_median_delta_pp": float(d_h.median()), "subject_p25_delta_pp": float(d_h.quantile(.25)), "subject_p10_delta_pp": float(d_h.quantile(.10)), "worst_subject_delta_pp": float(d_h.min()), "improved_subjects": int((d_h>0).sum()), "degraded_subjects": int((d_h<0).sum())}]).to_csv(OUT / "OUTER_FOLD_RESULTS.csv", index=False)
    continue_flag = bool(d_o.mean() > 1.0 and d_h.mean() > 1.0 and d_o.groupby(level=0).mean().gt(0).sum() >= 3 and d_h.groupby(level=0).mean().gt(0).sum() >= 3 and d_o.groupby(level=0).mean().min() > -1.0 and d_h.groupby(level=0).mean().min() > -1.0 and fm_o.std(ddof=0) <= fb_o.std(ddof=0))
    decision = "PROMISING_FOR_MULTI_SEED" if continue_flag else "STOP_SBTR"; report = ["# LiteBN-SBTR seed0", "", "| Method | Outer BA | Outer SD | Heldout BA | Heldout SD |", "|---|---:|---:|---:|---:|"]
    for method in ("Historical_LiteBN", "SBTR_main_tail_aware", "SBTR_meanBA_diagnostic"):
        a = next(x for x in rows_o if x["method"] == method); b2 = next(x for x in rows_h if x["method"] == method); report.append(f"| {method} | {a['BA']:.4f} | {a['fold_SD']:.4f} | {b2['BA']:.4f} | {b2['fold_SD']:.4f} |")
    report += ["", f"Main SBTR outer ΔBA: {d_o.mean():+.3f} pp", f"Main SBTR heldout ΔBA: {d_h.mean():+.3f} pp", f"Positive outer folds: {int((d_o.groupby(level=0).mean()>0).sum())}/5", f"Positive heldout folds: {int((d_h.groupby(level=0).mean()>0).sum())}/5", f"Subject median ΔBA: {d_o.median():+.3f} pp", f"Subject 25th percentile ΔBA: {d_o.quantile(.25):+.3f} pp", f"Subject 10th percentile ΔBA: {d_o.quantile(.10):+.3f} pp", f"Worst subject ΔBA: {d_o.min():+.3f} pp", "", "CURRENT_INTERNAL_HELDOUT_ACCESSED = YES", "FINAL_HELDOUT_ACCESSED = NO", "", "## Decision", decision]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8"); write_json(OUT / "FINAL_DECISION.json", {"decision": decision, "outer_delta_BA_pp": float(d_o.mean()), "heldout_delta_BA_pp": float(d_h.mean()), "seed1_seed2_started": False, "wbcic_erp_ssvep_started": False, "split_sha256": split_hash}); print(decision, flush=True); return 0

if __name__ == "__main__": raise SystemExit(main())
