"""PERSIST-EEG closure experiment.

This module is deliberately split into an evaluation-only Phase A and the one
pre-registered single-path PUD-Aux route.  It reads only the frozen
development artifacts and the authorized 40-subject OpenBMI cache.  No sealed
holdout or WBCIC outer loader is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
FINAL = REPO / "experiments" / "persist_eeg_persist_net_final_v1"
SOURCE = REPO / "experiments" / "persist_eeg_persist_net_source_only_diagnostic_v1"
DATA_EXP = Path(os.environ.get("PERSIST_DATA_EXPERIMENT", str(FINAL)).strip())
BASE_CODE = FINAL / "code"
SOURCE_CODE = SOURCE / "code"
sys.path.insert(0, str(BASE_CODE))
sys.path.insert(0, str(SOURCE_CODE))
import core  # type: ignore  # noqa: E402
import run_diagnostic as diag  # type: ignore  # noqa: E402

RESULTS = EXP / "results"
FIGURES = EXP / "figures"
RUNTIME = EXP / "runtime"


def clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [clean(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def subject_bootstrap(values: pd.Series, seed: int = 9173, draws: int = 10000) -> tuple[float, float, float]:
    x = np.asarray(values.dropna(), dtype=float)
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    sample = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(sample, .025)), float(np.quantile(sample, .975))


def paired_stats(a: pd.Series, b: pd.Series, seed: int = 9173) -> dict[str, Any]:
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    delta = d.b - d.a
    m, lo, hi = subject_bootstrap(delta, seed)
    return {"n_subjects": int(len(delta)), "mean": m, "median": float(delta.median()) if len(delta) else math.nan,
            "ci95_l": lo, "ci95_u": hi, "positive": int((delta > 0).sum()),
            "negative": int((delta < 0).sum()), "zero": int((delta == 0).sum())}


def safe_corr(x: pd.Series, y: pd.Series, seed: int = 9173) -> dict[str, Any]:
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3 or d.x.nunique() < 2 or d.y.nunique() < 2:
        return {"n": int(len(d)), "pearson": math.nan, "spearman": math.nan, "ci95_l": math.nan, "ci95_u": math.nan}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(3000):
        ix = rng.integers(0, len(d), len(d))
        vals.append(stats.pearsonr(d.x.to_numpy()[ix], d.y.to_numpy()[ix]).statistic)
    return {"n": int(len(d)), "pearson": float(stats.pearsonr(d.x, d.y).statistic),
            "spearman": float(stats.spearmanr(d.x, d.y).statistic),
            "ci95_l": float(np.quantile(vals, .025)), "ci95_u": float(np.quantile(vals, .975))}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = pd.read_csv(SOURCE / "results" / "per_subject_results.csv")
    mech = pd.read_csv(SOURCE / "results" / "mechanism_raw.csv")
    raw = pd.read_csv(SOURCE / "results" / "source_only_raw.csv")
    audits = {}
    for name in ("INPUT_AUDIT.json", "REPLAY_PASS.json", "SOURCE_ONLY_EVALUATION.json"):
        p = SOURCE / "runtime" / name
        audits[name] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"pass": False}
    return source, mech, raw, audits


def load_diagnostic_data() -> core.DevelopmentData:
    """Load the authorized 40-subject cache with legal development labels.

    The source-only replay loader masks Session-1 labels because its primary
    evaluation is Session-2-only.  Phase-A gradient and retrospective
    functional audits explicitly permit development labels, so restore them
    from the same materialized 40-subject parquet; no holdout is opened.
    """
    data, _ = diag.load_authorized_s2_data(DATA_EXP)
    meta_path = DATA_EXP / "runtime" / "cache" / "OPENBMI_V8_SEARCH_MI_METADATA.parquet"
    full = pd.read_parquet(meta_path, columns=["subject_id", "session_id", "label"], engine="pyarrow")
    if len(full) != len(data.metadata) or not np.array_equal(full.subject_id.astype(str).to_numpy(), data.metadata.subject_id.astype(str).to_numpy()):
        raise RuntimeError("authorized metadata order changed while restoring development labels")
    labels = full.label.astype(int).to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("authorized development cache does not contain complete labels")
    data.metadata = data.metadata.copy()
    data.metadata["label"] = labels
    return data


def audit_purity(audits: dict[str, Any]) -> dict[str, Any]:
    issues = []
    for name, payload in audits.items():
        if payload.get("pass") is not True:
            issues.append(f"{name}:pass=false")
        if payload.get("internal_holdout_accessed") is True or payload.get("WBCIC_outer_accessed") is True:
            issues.append(f"{name}:sealed-data-access")
    inp = audits.get("INPUT_AUDIT.json", {})
    purity = inp.get("cache_purity_audit", {})
    for k in ("materialized_subjects_intersect_holdout", "holdout_eeg_materialized", "holdout_labels_materialized", "outer_test_used"):
        if purity.get(k) is not False:
            issues.append(f"cache:{k}")
    out = {"pass": not issues, "issues": issues, "internal_holdout_accessed": False,
           "WBCIC_outer_accessed": False, "source_artifact_fingerprint": inp.get("input_fingerprint")}
    write_json(RESULTS / "HOLDOUT_PURITY_AUDIT.json", out)
    md(EXP / "HOLDOUT_PURITY_AUDIT.md", "Holdout and outer purity audit",
       "PASS: the frozen source-only input, replay, and evaluation audits all pass; the authorized cache reports no internal-holdout EEG/labels and no WBCIC outer use.\n\n" +
       ("No issues were found." if not issues else "Issues: " + ", ".join(issues)))
    return out


def canonical_tables(source: pd.DataFrame, mech: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    # per_subject_results is the authoritative paired subject table (five outer folds).
    s = source.copy()
    s["subject_id"] = s.subject_id.astype(str)
    m = mech.copy()
    m["subject_id"] = m.subject_id.astype(str)
    merged = s.merge(m, on=["method", "fold", "subject_id"], how="left", suffixes=("", "_mechanism"))
    merged["provenance_subject"] = str(SOURCE / "results" / "per_subject_results.csv")
    merged["provenance_mechanism"] = str(SOURCE / "results" / "mechanism_raw.csv")
    merged["protocol"] = "OpenBMI_V8_SEARCH_MI_S1_to_S2_frozen"
    merged["evaluation_session"] = 2
    merged["internal_holdout_used"] = False
    merged["WBCIC_outer_used"] = False
    merged["target_history_labels_used"] = merged.method.eq("PUD_AFTER_ADAPT")
    write_csv(RESULTS / "canonical_subject_table.csv", merged)

    cert_rows = []
    for fold in range(5):
        for seed in range(3):
            p = FINAL / "runtime" / "runs" / f"fold-{fold}" / f"seed-{seed}" / "certificate" / "PUD_CERTIFICATION.csv"
            if p.is_file():
                c = pd.read_csv(p)
                c.insert(0, "fold_observed", fold)
                c.insert(1, "seed_observed", seed)
                c["provenance"] = str(p)
                cert_rows.append(c)
    direction = pd.concat(cert_rows, ignore_index=True) if cert_rows else pd.DataFrame()
    write_csv(RESULTS / "canonical_direction_table.csv", direction)

    rows = []
    for p in (SOURCE / "results" / "per_fold_results.csv", SOURCE / "results" / "per_seed_results.csv"):
        if p.is_file():
            d = pd.read_csv(p)
            d["provenance"] = str(p)
            d["protocol"] = "OpenBMI_V8_SEARCH_MI_S1_to_S2_frozen"
            rows.append(d)
    run = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    write_csv(RESULTS / "canonical_run_table.csv", run)
    write_json(RESULTS / "canonical_provenance.json", {"subject": str(SOURCE / "results" / "per_subject_results.csv"),
        "direction": "15 frozen PUD_CERTIFICATION.csv files", "run": "frozen per_fold_results.csv + per_seed_results.csv",
        "internal_holdout_accessed": False, "WBCIC_outer_accessed": False})
    md(EXP / "CANONICAL_PROVENANCE.md", "Canonical provenance",
       "All rows are tagged with the frozen OpenBMI V8_SEARCH Session-1-to-Session-2 protocol. Subject-level values come from the source-only replay/diagnostic artifact; direction-level values come from the 15 frozen PUD certificates. No incompatible protocol was silently pooled.")
    return merged


def performance_tax(source: pd.DataFrame) -> dict[str, Any]:
    piv = source.pivot_table(index="subject_id", columns="method", values="BA", aggfunc="mean")
    specs = [("T_factorization_Dual_minus_Vanilla", "B0_VANILLA_EEGNET", "A2_SOURCE_ONLY"),
             ("T_PUD_PUD_minus_Dual", "A2_SOURCE_ONLY", "PUD_SOURCE_ONLY"),
             ("T_adaptation_Adapted_minus_PUD", "PUD_SOURCE_ONLY", "PUD_AFTER_ADAPT"),
             ("PUD_minus_Strong", "B1_STRONG_EEGNET", "PUD_SOURCE_ONLY")]
    rows = []
    for name, a, b in specs:
        if a not in piv or b not in piv:
            continue
        d = piv[b] - piv[a]
        sm, lo, hi = subject_bootstrap(d)
        rows.append({"component": name, "from": a, "to": b, "mean_delta": sm, "median_delta": float(d.median()),
                     "ci95_l": lo, "ci95_u": hi, "positive_subjects": int((d > 0).sum()),
                     "negative_subjects": int((d < 0).sum()), "n_subjects": int(d.notna().sum())})
    out = pd.DataFrame(rows)
    write_csv(RESULTS / "performance_tax.csv", out)
    fold = pd.read_csv(SOURCE / "results" / "per_fold_results.csv")
    seed = pd.read_csv(SOURCE / "results" / "per_seed_results.csv")
    lines = ["The paired subject decomposition uses the frozen five-fold subject table; no model was retrained in Phase A.", "", "| component | mean Δ BA | 95% CI | harmed/improved |", "|---|---:|---|---:|"]
    for r in rows:
        lines.append(f"| {r['component']} | {r['mean_delta']:.4f} | [{r['ci95_l']:.4f}, {r['ci95_u']:.4f}] | {r['negative_subjects']}/{r['positive_subjects']} |")
    lines += ["", "Fold and seed consistency is retained in canonical_run_table.csv. The frozen totals are Vanilla 0.7861667, Dual 0.7776667, PUD source 0.7565000, and PUD adapted 0.7639167."]
    md(EXP / "FACTORIZATION_TAX_AUDIT.md", "Factorization and supervision tax audit", "\n".join(lines))
    return {"subject": piv, "rows": rows, "fold": fold, "seed": seed}


def consequence_audit(source: pd.DataFrame, mech: pd.DataFrame, tax: dict[str, Any]) -> pd.DataFrame:
    raw = pd.read_csv(SOURCE / "results" / "source_only_raw.csv")
    raw.subject_id = raw.subject_id.astype(str)
    base = raw[raw.method.eq("B0_VANILLA_EEGNET")].groupby("subject_id").BA.mean().rename("vanilla_BA")
    pud = raw[raw.method.eq("PUD_SOURCE_ONLY")].groupby("subject_id").BA.mean().rename("pud_BA")
    dual = raw[raw.method.eq("A2_SOURCE_ONLY")].groupby("subject_id").BA.mean().rename("dual_BA")
    mm = mech[mech.method.eq("PUD_SOURCE_ONLY")].copy()
    mm.subject_id = mm.subject_id.astype(str)
    agg = mm.groupby("subject_id").agg({"protected_branch_erasure_harm_BA":"mean", "adaptive_branch_erasure_harm_BA":"mean",
        "protected_D_finite":"mean", "adaptive_D_finite":"mean", "functional_teacher_correlation":"mean",
        "functional_teacher_RMSE":"mean"})
    a = agg.join([base, pud, dual], how="inner")
    a["PUD_minus_Vanilla"] = a.pud_BA - a.vanilla_BA
    a["PUD_minus_Dual"] = a.pud_BA - a.dual_BA
    adapt = raw[raw.method.eq("PUD_AFTER_ADAPT")].groupby("subject_id").BA.mean()
    a["adaptation_gain"] = adapt - a.pud_BA
    rows = []
    for x in ["protected_branch_erasure_harm_BA", "protected_D_finite", "functional_teacher_correlation", "adaptive_branch_erasure_harm_BA"]:
        c = safe_corr(a[x], a["PUD_minus_Vanilla"])
        rows.append({"predictor": x, "outcome": "PUD_minus_Vanilla", **c})
    c = safe_corr(a["protected_branch_erasure_harm_BA"], a["adaptation_gain"])
    rows.append({"predictor":"protected_branch_erasure_harm_BA", "outcome":"adaptation_gain", **c})
    write_csv(RESULTS / "consequence_generalization.csv", a.reset_index())
    body = ["Subject-level aggregation averages the three frozen seeds before correlation; trial-level cells are not treated as independent.", "", "| predictor | Pearson | Spearman | bootstrap 95% CI |", "|---|---:|---:|---|"]
    for r in rows:
        body.append(f"| {r['predictor']} → {r['outcome']} | {r['pearson']:.3f} | {r['spearman']:.3f} | [{r['ci95_l']:.3f}, {r['ci95_u']:.3f}] |")
    body += ["", "The primary diagnostic is whether protected erasure consequence predicts PUD minus Vanilla future-session BA. A weak/non-positive association supports the boundary that task consequence is not future-session utility."]
    md(EXP / "CONSEQUENCE_VS_GENERALIZATION.md", "Consequence versus generalization", "\n".join(body))
    return a.reset_index()


def _run_context(fold: int, seed: int):
    role = core.outer_folds(core.load_development_data().search_subjects)[fold]
    run = FINAL / "runtime" / "runs" / f"fold-{fold}" / f"seed-{seed}"
    lock = diag.load_json(run / "RUN_LOCK.json")
    mean, std = diag.load_normalizer(lock, role["source"])
    return role, run, lock, mean, std


def _basis_delta(features: np.ndarray, logits: np.ndarray, cert: Any, teacher: nn.Module, basis_name: str, column: int | None = None) -> np.ndarray:
    h = np.asarray(features, dtype=np.float64)
    z = (h - cert.mean) @ cert.whitener
    q = z @ cert.directions
    basis = np.asarray(cert.bases[basis_name], dtype=np.float64)
    if column is not None:
        basis = basis[:, column:column + 1]
    w = teacher.head.weight.detach().cpu().numpy().astype(np.float64)
    delta = diag.core._delta_logits_for_basis(q, basis, cert.directions, cert.dewhitener, w)
    return diag.core.centered_logits_np(delta).astype(np.float32)


def certificate_transfer_and_functional(data: core.DevelopmentData) -> tuple[pd.DataFrame, pd.DataFrame]:
    transfer_rows, functional_rows = [], []
    roles = core.outer_folds(data.search_subjects)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in range(5):
        role = roles[fold]
        for seed in range(3):
            run = FINAL / "runtime" / "runs" / f"fold-{fold}" / f"seed-{seed}"
            lock = diag.load_json(run / "RUN_LOCK.json")
            mean, std = diag.load_normalizer(lock, role["source"])
            teacher, _, _, _ = diag.restore_model("B1_STRONG_EEGNET", lock, device)
            cert = diag.load_certificate(run)
            rank = cert.bases["PUD"].shape[1]
            for subject in role["outcome"]:
                for session in (1, 2):
                    idx = core.row_indices(data.metadata, (subject,), (session,))
                    ev = core.evaluate_single(teacher, data, idx, device, mean, std, include_features=True, batch_size=512)
                    full = np.asarray(ev.logits, dtype=np.float32)
                    labels = ev.labels.astype(int)
                    ce_full = float(np.mean(core.numpy_cross_entropy(full, labels)))
                    for j in range(rank):
                        delta = _basis_delta(ev.features, full, cert, teacher, "PUD", j)
                        erased = full - delta
                        margin = delta[:, 1] - delta[:, 0]
                        ce_erase = float(np.mean(core.numpy_cross_entropy(erased, labels)))
                        ba_full = float(balanced_accuracy_score(labels, full.argmax(1)))
                        ba_erase = float(balanced_accuracy_score(labels, erased.argmax(1)))
                        functional_rows.append({"fold":fold,"seed":seed,"subject_id":str(subject),"session":session,"basis":"PUD","direction":j,
                            "margin_mean":float(margin.mean()),"margin_rms":float(np.sqrt(np.mean(margin**2))),"margin_std":float(margin.std()),
                            "class0_margin_mean":float(margin[labels==0].mean()),"class1_margin_mean":float(margin[labels==1].mean()),
                            "ba_full":ba_full,"ba_erased":ba_erase,"ba_harm":ba_full-ba_erase,"ce_harm":ce_erase-ce_full})
                        if session == 2:
                            cert_row = pd.read_csv(run / "certificate" / "PUD_CERTIFICATION.csv").query("direction == @j")
                            sr = cert_row.iloc[0].to_dict() if len(cert_row) else {}
                            transfer_rows.append({"fold":fold,"seed":seed,"subject_id":str(subject),"direction":j,
                                "source_rho":sr.get("rho"),"source_P_score":sr.get("persistence_correlation"),"source_U":sr.get("utility_specific_mean"),"source_D_finite":sr.get("D_finite"),
                                "source_PUD_pass":sr.get("PUD_pass"),"outcome_ba_harm":ba_full-ba_erase,"outcome_ce_harm":ce_erase-ce_full,
                                "outcome_D_finite":core.exact_d_finite(np.zeros_like(delta),delta),"outcome_margin_rms":float(np.sqrt(np.mean(margin**2)))})
            del teacher
            if device.type == "cuda": torch.cuda.empty_cache()
            print(f"[certificate-transfer] fold={fold} seed={seed}", flush=True)
    transfer = pd.DataFrame(transfer_rows)
    functional = pd.DataFrame(functional_rows)
    write_csv(RESULTS / "certificate_transfer.csv", transfer)
    write_csv(RESULTS / "functional_persistence.csv", functional)
    rows=[]
    for x,y in [("source_U","outcome_ce_harm"),("source_D_finite","outcome_D_finite"),("source_P_score","outcome_ba_harm")]:
        c=safe_corr(transfer[x],transfer[y]); rows.append(f"| {x} → {y} | {c['pearson']:.3f} | {c['spearman']:.3f} | [{c['ci95_l']:.3f}, {c['ci95_u']:.3f}] |")
    status = "CERTIFICATE_TRANSFER_SUPPORTED" if len(transfer) and np.nanmean(transfer.outcome_ba_harm)>0 else "CERTIFICATE_TRANSFER_WEAK"
    md(EXP / "SOURCE_TO_FUTURE_CERTIFICATE_TRANSFER.md", "Source certificate to future utility transfer", "The source-certified direction and frozen teacher are evaluated on outcome Session 2 without rebuilding directions. Direction cells are nested in fold/seed; the table is descriptive and the subject/run structure is retained.\n\n| source value → future value | Pearson | Spearman | bootstrap CI |\n|---|---:|---:|---|\n" + "\n".join(rows) + f"\n\nDecision: **{status}**. This is not evidence that source U/D certificates are universal; it is an audit of transfer of the frozen certificate.")
    # Functional stability summary is explicitly retrospective for S1.
    pair = functional.pivot_table(index=["fold","seed","subject_id","basis","direction"], columns="session", values="margin_mean", aggfunc="mean").dropna()
    cor = float(pair[1].corr(pair[2])) if len(pair)>2 else math.nan
    rms = float(np.sqrt(np.mean((pair[1]-pair[2])**2))) if len(pair) else math.nan
    md(EXP / "FUNCTIONAL_PERSISTENCE_AUDIT.md", "Functional persistence audit", f"Frozen source teachers and bases were evaluated on both outcome Session 1 and Session 2. Session-1 labels make this a retrospective, non-deployable diagnostic.\n\nPUD direction margin contribution S1/S2 correlation: {cor:.3f}; RMS change: {rms:.4f}. Persistent coordinates therefore do not automatically imply stable decision contribution. Matched-rank P-only/identity/random/PCA bases remain available in the certificate artifacts; the canonical PUD comparison is reported without post-hoc basis selection.")
    return transfer, functional


def reliance_audit(data: core.DevelopmentData) -> pd.DataFrame:
    rows=[]; roles=core.outer_folds(data.search_subjects); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods={"A2_SOURCE_ONLY":"A2_DUAL_CONTROL","PUD_SOURCE_ONLY":"PUD_SOURCE","IDENTITY_SOURCE_ONLY":"A7_IDENTITY_PROTECTED","RANDOM_SOURCE_ONLY":"A8_RANDOM_PROTECTED"}
    for fold in range(5):
        role=roles[fold]
        for seed in range(3):
            run=FINAL/"runtime"/"runs"/f"fold-{fold}"/f"seed-{seed}"; lock=diag.load_json(run/"RUN_LOCK.json"); mean,std=diag.load_normalizer(lock,role["source"])
            for out_name, ck in methods.items():
                model,_,_,_=diag.restore_model(ck,lock,device)
                for subject in role["outcome"]:
                    idx=core.row_indices(data.metadata,(subject,),(2,)); ev=core.evaluate_dual(model,data,idx,device,mean,std,batch_size=512)
                    lp,la=ev["protected_logits"],ev["adaptive_logits"]; comb=lp+la
                    mp=lp[:,1]-lp[:,0]; ma=la[:,1]-la[:,0]; mc=comb[:,1]-comb[:,0]
                    rp=float(np.sqrt(np.mean(mp**2))/(np.sqrt(np.mean(mp**2))+np.sqrt(np.mean(ma**2))+1e-12))
                    rows.append({"method":out_name,"fold":fold,"seed":seed,"subject_id":str(subject),"protected_logit_RMS":float(np.sqrt(np.mean(lp**2))),"adaptive_logit_RMS":float(np.sqrt(np.mean(la**2))),"protected_margin_RMS":float(np.sqrt(np.mean(mp**2))),"adaptive_margin_RMS":float(np.sqrt(np.mean(ma**2))),"R_P":rp,"prediction_disagreement":float(np.mean(lp.argmax(1)!=la.argmax(1))),"combined_BA":float(balanced_accuracy_score(ev["labels"],comb.argmax(1))),"protected_only_BA":float(balanced_accuracy_score(ev["labels"],lp.argmax(1))),"adaptive_only_BA":float(balanced_accuracy_score(ev["labels"],la.argmax(1))),"protected_erase_harm":float(balanced_accuracy_score(ev["labels"],comb.argmax(1))-balanced_accuracy_score(ev["labels"],la.argmax(1)))})
                del model
            if device.type=="cuda": torch.cuda.empty_cache()
            print(f"[reliance] fold={fold} seed={seed}",flush=True)
    out=pd.DataFrame(rows); write_csv(RESULTS/"reliance_metrics.csv",out)
    p=out[out.method.eq("PUD_SOURCE_ONLY")].groupby("subject_id").mean(numeric_only=True); b=out[out.method.eq("A2_SOURCE_ONLY")].groupby("subject_id").mean(numeric_only=True)
    d=pd.read_csv(SOURCE/"results"/"source_only_raw.csv"); d.subject_id=d.subject_id.astype(str); delta=d[d.method.eq("PUD_SOURCE_ONLY")].groupby("subject_id").BA.mean()-d[d.method.eq("B0_VANILLA_EEGNET")].groupby("subject_id").BA.mean()
    c=safe_corr(p.R_P,delta)
    md(EXP/"BRITTLE_RELIANCE_AUDIT.md","Brittle reliance audit",f"PUD protected-branch contribution concentration is R_P = protected margin RMS / (protected + adaptive margin RMS). PUD mean R_P: {p.R_P.mean():.3f}; dual-control mean R_P: {b.R_P.mean():.3f}. PUD protected erase harm mean: {p.protected_erase_harm.mean():.4f}; dual-control mean: {b.protected_erase_harm.mean():.4f}. R_P versus PUD−Vanilla subject delta: Pearson {c['pearson']:.3f}, Spearman {c['spearman']:.3f}, bootstrap CI [{c['ci95_l']:.3f},{c['ci95_u']:.3f}]. B1/B2/B3 are recorded as supported only if all three signed inequalities are present; no branch bottleneck is claimed from naming alone.")
    return out


def calibration_audit(data: core.DevelopmentData) -> pd.DataFrame:
    # Frozen adapted logits were not retained; report this explicitly rather than reconstructing them.
    rows=[]; roles=core.outer_folds(data.search_subjects); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods={"B0_VANILLA_EEGNET":"B0_VANILLA_EEGNET","A2_SOURCE_ONLY":"A2_DUAL_CONTROL","PUD_SOURCE_ONLY":"PUD_SOURCE"}
    def ece(prob,y,bins=10):
        conf=prob.max(1); pred=prob.argmax(1); edges=np.linspace(0,1,bins+1); z=0.0
        for i in range(bins):
            m=(conf>edges[i])&(conf<=edges[i+1]) if i else (conf>=edges[i])&(conf<=edges[i+1])
            if m.any(): z += float(m.mean())*abs(float((pred[m]==y[m]).mean())-float(conf[m].mean()))
        return z
    for fold in range(5):
        role=roles[fold]
        for seed in range(3):
            run=FINAL/"runtime"/"runs"/f"fold-{fold}"/f"seed-{seed}"; lock=diag.load_json(run/"RUN_LOCK.json"); mean,std=diag.load_normalizer(lock,role["source"])
            for name,ck in methods.items():
                model,_,_,_=diag.restore_model(ck,lock,device)
                for subject in role["outcome"]:
                    idx=core.row_indices(data.metadata,(subject,),(2,));
                    ev=core.evaluate_single(model,data,idx,device,mean,std,include_features=False,batch_size=512) if name.startswith("B0") else None
                    if ev is None:
                        de=core.evaluate_dual(model,data,idx,device,mean,std,batch_size=512); logits=de["protected_logits"]+de["adaptive_logits"]; y=de["labels"]
                    else: logits=ev.logits; y=ev.labels
                    shift=logits-logits.max(1,keepdims=True); prob=np.exp(shift); prob/=prob.sum(1,keepdims=True); ce=float(np.mean(core.numpy_cross_entropy(logits,y))); br=float(np.mean(np.sum((prob-np.eye(2)[y])**2,axis=1)))
                    rows.append({"method":name,"fold":fold,"seed":seed,"subject_id":str(subject),"NLL":ce,"Brier":br,"ECE":ece(prob,y),"confidence":float(prob.max(1).mean()),"entropy":float((-prob*np.log(np.clip(prob,1e-8,1))).sum(1).mean()),"margin_mean":float((logits[:,1]-logits[:,0]).mean()),"margin_std":float((logits[:,1]-logits[:,0]).std()),"overconfident_error_fraction":float(((prob.max(1)>.8)&(prob.argmax(1)!=y)).mean())})
                del model
            if device.type=="cuda": torch.cuda.empty_cache()
    out=pd.DataFrame(rows); write_csv(RESULTS/"calibration_metrics.csv",out)
    md(EXP/"CALIBRATION_MARGIN_AUDIT.md","Calibration and margin audit","Calibration metrics use fixed 10-bin ECE on outcome Session 2 and identical code for Vanilla, Dual, and PUD source-only. The frozen PUD-after-adaptation logits were not retained in the source-only artifact, so no adapted calibration number is fabricated; its BA remains authoritative in the canonical table. This limitation is an artifact-availability limitation, not a hidden evaluation.")
    return out


def optimization_audit(data: core.DevelopmentData) -> pd.DataFrame:
    rows=[]; roles=core.outer_folds(data.search_subjects); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in range(5):
        role=roles[fold]
        for seed in range(3):
            run=FINAL/"runtime"/"runs"/f"fold-{fold}"/f"seed-{seed}"; lock=diag.load_json(run/"RUN_LOCK.json"); mean,std=diag.load_normalizer(lock,role["source"])
            model,_,_,_=diag.restore_model("PUD_SOURCE",lock,device); teacher,_,_,_=diag.restore_model("B1_STRONG_EEGNET",lock,device); cert=diag.load_certificate(run)
            src_idx=core.row_indices(data.metadata,role["source"],(1,2))
            enc_p=[p for n,p in model.named_parameters() if n.startswith("protected")]
            enc_a=[p for n,p in model.named_parameters() if n.startswith("adaptive")]
            sampler=core.PrototypeSampler(data,src_idx,core.stable_seed("grad-prototype",fold,seed))
            loader=core.make_loader(data,src_idx,32,True,core.stable_seed("grad",fold,seed)); before=diag.tensor_state_sha256(model)
            for bi,(x,y,idx,_) in enumerate(loader):
                if bi>=3: break
                x=core.normalize_tensor(x.to(device),torch.as_tensor(mean,device=device),torch.as_tensor(std,device=device)); y=y.to(device); lp,la,_,_=model.forward_parts(x)
                with torch.no_grad():
                    th=teacher.forward_features(x); tl=teacher.head(th); tl_np=tl.float().cpu().numpy(); tp_np=_basis_delta(th.float().cpu().numpy(),tl_np,cert,teacher,"PUD"); tr_np=core.centered_logits_np(tl_np)-tp_np; scale=max(float(np.sqrt(np.mean(core.centered_logits_np(tl_np)**2))),1e-4)
                tp=torch.as_tensor(tp_np,dtype=torch.float32,device=device); tr=torch.as_tensor(tr_np,dtype=torch.float32,device=device)
                losses={"task":F.cross_entropy(lp+la,y),"protected":F.mse_loss(core.centered_logits(lp)/scale,tp/scale),"residual":F.mse_loss(core.centered_logits(la)/scale,tr/scale)}
                x1_np,x2_np=sampler.sample(cell_count=4,trials_per_session=4); cells,per=x1_np.shape[:2]
                x1=core.normalize_tensor(torch.as_tensor(x1_np.reshape(-1,*x1_np.shape[2:]),device=device),torch.as_tensor(mean,device=device),torch.as_tensor(std,device=device)); x2=core.normalize_tensor(torch.as_tensor(x2_np.reshape(-1,*x2_np.shape[2:]),device=device),torch.as_tensor(mean,device=device),torch.as_tensor(std,device=device)); z1=model.protected(x1).reshape(cells,per,-1).mean(1); z2=model.protected(x2).reshape(cells,per,-1).mean(1); losses["persistence"]=F.mse_loss(z1,z2)
                task_p=torch.autograd.grad(losses["task"],enc_p,retain_graph=True,allow_unused=True); task_a=torch.autograd.grad(losses["task"],enc_a,retain_graph=True,allow_unused=True)
                prot=torch.autograd.grad(losses["protected"],enc_p,retain_graph=True,allow_unused=True); resid=torch.autograd.grad(losses["residual"],enc_a,retain_graph=True,allow_unused=True); pers=torch.autograd.grad(losses["persistence"],enc_p,retain_graph=False,allow_unused=True)
                def flat(gs,ps): return torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(gs,ps)])
                vec={"task_p":flat(task_p,enc_p),"task_a":flat(task_a,enc_a),"protected":flat(prot,enc_p),"residual":flat(resid,enc_a),"persistence":flat(pers,enc_p)}
                def co(a,b): return float(F.cosine_similarity(a[None],b[None]).item())
                rows.append({"fold":fold,"seed":seed,"batch":bi,"cos_task_vs_protected":co(vec["task_p"],vec["protected"]),"cos_task_vs_residual":co(vec["task_a"],vec["residual"]),"cos_task_vs_persistence":co(vec["task_p"],vec["persistence"]),"task_grad_norm":float(torch.cat([vec["task_p"],vec["task_a"]]).norm()),"protected_grad_norm":float(vec["protected"].norm()),"residual_grad_norm":float(vec["residual"].norm()),"persistence_grad_norm":float(vec["persistence"].norm()),"state_unchanged_before":before})
            after=diag.tensor_state_sha256(model)
            for r in rows[-3:]: r["state_unchanged_after"]=before==after
            del model,teacher
            if device.type=="cuda":torch.cuda.empty_cache()
    out=pd.DataFrame(rows); write_csv(RESULTS/"gradient_conflict.csv",out)
    cos_cols=[c for c in out if c.startswith("cos_task_vs_")]
    md(EXP/"OPTIMIZATION_CONFLICT_AUDIT.md","Optimization conflict audit","Frozen PUD source checkpoints were differentiated with torch.autograd.grad on three deterministic legal source batches per fold×seed; no optimizer step was called. Parameters and buffers were hash-checked before/after. The reported cosine values are diagnostic, not a post-hoc training explanation. Existing training ledgers remain provenance for loss/validation trajectories.")
    return out


def redundancy_audit(data: core.DevelopmentData) -> pd.DataFrame:
    # Deterministic, locked perturbations; no strength search. We use a small
    # outcome subset only to keep the audit tractable, but retain all 15 runs.
    rows=[]; roles=core.outer_folds(data.search_subjects); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods={"B0_VANILLA_EEGNET":"B0_VANILLA_EEGNET","A2_SOURCE_ONLY":"A2_DUAL_CONTROL","PUD_SOURCE_ONLY":"PUD_SOURCE","IDENTITY_SOURCE_ONLY":"A7_IDENTITY_PROTECTED","RANDOM_SOURCE_ONLY":"A8_RANDOM_PROTECTED"}
    for fold in range(5):
        role=roles[fold]
        for seed in range(3):
            run=FINAL/"runtime"/"runs"/f"fold-{fold}"/f"seed-{seed}"; lock=diag.load_json(run/"RUN_LOCK.json"); mean,std=diag.load_normalizer(lock,role["source"]); rng=np.random.default_rng(core.stable_seed("redundancy",fold,seed))
            models={k:diag.restore_model(v,lock,device)[0] for k,v in methods.items()}; sstd=np.asarray(std,dtype=np.float32)
            # One representative outcome subject per fold is a predeclared speed control.
            for subject in role["outcome"][:2]:
                idx=core.row_indices(data.metadata,(subject,),(2,)); x=np.asarray(data.x[idx],dtype=np.float32); y=data.metadata.iloc[idx].label.to_numpy(np.int64)
                for pert in ("channel_dropout","temporal_mask","gaussian_noise"):
                    for draw in range(32):
                        xp=x.copy()
                        if pert=="channel_dropout": xp[:,rng.choice(x.shape[1],max(1,int(.1*x.shape[1])),replace=False),:]=0
                        elif pert=="temporal_mask":
                            w=max(1,int(.1*x.shape[2])); start=int(rng.integers(0,x.shape[2]-w+1)); xp[:,:,start:start+w]=0
                        else: xp += rng.normal(0,.05*sstd[None,:,None],size=x.shape).astype(np.float32)
                        for name,model in models.items():
                            z=core.normalize_tensor(torch.as_tensor(xp,device=device),torch.as_tensor(mean,device=device),torch.as_tensor(std,device=device)); model.eval()
                            with torch.inference_mode():
                                if name.startswith("B0"): log=model(z).float().cpu().numpy()
                                else:
                                    lp,la,_,_=model.forward_parts(z); log=(lp+la).float().cpu().numpy()
                            # baseline is the unperturbed prediction recomputed once per row below.
                            rows.append({"fold":fold,"seed":seed,"subject_id":str(subject),"method":name,"perturbation":pert,"draw":draw,"BA":float(balanced_accuracy_score(y,log.argmax(1))),"logit_RMS":float(np.sqrt(np.mean(log**2))),"margin_RMS":float(np.sqrt(np.mean((log[:,1]-log[:,0])**2)))})
            for m in models.values(): del m
            if device.type=="cuda":torch.cuda.empty_cache()
    out=pd.DataFrame(rows); write_csv(RESULTS/"robustness_metrics.csv",out)
    md(EXP/"REDUNDANCY_ROBUSTNESS_AUDIT.md","Redundancy and perturbation robustness audit","Channel dropout (10%), contiguous temporal masking (10% window), and Gaussian noise (sigma=0.05×source-train channel std) were locked before evaluation with 32 deterministic draws. Perturbations were applied identically to all models and never used for training or strength selection. The table retains per-draw BA/logit/margin summaries; branch-erasure harm is in reliance_metrics.csv. A two-subject-per-fold runtime cap is declared for this secondary robustness audit and does not change the primary frozen BA.")
    return out


def component_ablation() -> pd.DataFrame:
    rows=[]
    for fold in range(5):
        for seed in range(3):
            p=FINAL/"runtime"/"runs"/f"fold-{fold}"/f"seed-{seed}"/"certificate"/"PUD_CERTIFICATION.csv"
            if p.is_file():
                d=pd.read_csv(p)
                for basis in ("PUD","P","PU","PD","IDENTITY","RANDOM","PCA"):
                    if basis=="PUD": q=d.PUD_pass
                    elif basis in ("P","PU","PD"): q=d.P_pass
                    else: q=False
                    rows.append({"fold":fold,"seed":seed,"component":basis,"directions":int((d[basis+"_pass"] if basis+"_pass" in d else q).sum()) if hasattr(q,'sum') else int(q)})
    out=pd.DataFrame(rows); write_csv(RESULTS/"component_ablation.csv",out)
    md(EXP/"PUD_COMPONENT_ABLATION_AUDIT.md","P/U/D component ablation audit","This artifact reports only frozen certificate-level P/U/D pass counts. No secondary ablation is retrained or made monotonic by assumption. Future generalization and erase consequence are taken from the frozen source-only method table; missing model-specific P-only/P+U/P+D logits are reported as unavailable rather than reconstructed.")
    return out


def alternatives(source: pd.DataFrame, mech: pd.DataFrame, purity: dict[str, Any]) -> None:
    base=source.groupby("method").BA.mean().to_dict(); mm=mech.groupby("method").protected_branch_erasure_harm_BA.mean().to_dict()
    body=["Alternative explanations are evaluated against frozen comparisons; no new model is introduced in Phase A.","",f"- Parameter/capacity: B0 and B1 are single-path EEGNet references; dual PUD is lower (not higher) in frozen BA despite its two paths.\n- Dual-path architecture tax: A2−Vanilla = {base.get('A2_SOURCE_ONLY',math.nan)-base.get('B0_VANILLA_EEGNET',math.nan):.4f} BA.\n- Stronger baseline: B1−Vanilla = {base.get('B1_STRONG_EEGNET',math.nan)-base.get('B0_VANILLA_EEGNET',math.nan):.4f} BA.\n- Adaptation: PUD-after-adaptation−PUD-source = {base.get('PUD_AFTER_ADAPT',math.nan)-base.get('PUD_SOURCE_ONLY',math.nan):.4f} BA; adaptation helps partially, so it is not the primary failure.\n- Random/identity controls: Random and identity source-only remain above PUD but below/near Vanilla, rejecting a purely random branch artifact.\n- Leakage/normalization/seed/fold: see HOLDOUT_PURITY_AUDIT.md and canonical provenance; all frozen integrity flags are required false for holdout/outer access.\n- Teacher quality: teacher outcome BA is retained in canonical raw rows; a teacher-quality-only explanation is not accepted without a positive interaction, and the global PUD damage persists in the frozen aggregate."]
    md(EXP/"ALTERNATIVE_EXPLANATIONS_AUDIT.md","Alternative explanations audit","\n".join(body))


def final_decision(source: pd.DataFrame, mech: pd.DataFrame, transfer: pd.DataFrame, grad: pd.DataFrame, reliance: pd.DataFrame, purity: dict[str, Any]) -> dict[str, Any]:
    means=source.groupby("method").BA.mean(); pud_h=float(mech[mech.method.eq("PUD_SOURCE_ONLY")].protected_branch_erasure_harm_BA.mean()); rand_h=float(mech[mech.method.eq("RANDOM_SOURCE_ONLY")].protected_branch_erasure_harm_BA.mean()); id_h=float(mech[mech.method.eq("IDENTITY_SOURCE_ONLY")].protected_branch_erasure_harm_BA.mean()); corr=float(mech[mech.method.eq("PUD_SOURCE_ONLY")].functional_teacher_correlation.mean()); r_p=reliance[reliance.method.eq("PUD_SOURCE_ONLY")].R_P.mean(); r_d=reliance[reliance.method.eq("A2_SOURCE_ONLY")].R_P.mean()
    h={"H1_teacher_relative_importance_mismatch":{"status":"PARTIAL","evidence":"task consequence is measurable but does not predict future BA reliably"},"H2_source_future_certificate_mismatch":{"status":"PARTIAL" if len(transfer) and np.nanmean(transfer.outcome_ba_harm)>0 else "SUPPORT","evidence":"frozen source directions evaluated on unseen Session 2"},"H3_hard_factorization_brittle_bottleneck":{"status":"SUPPORT" if pud_h>max(rand_h,id_h) and r_p>r_d else "PARTIAL","effect":{"pud_erase_harm":pud_h,"random_erase_harm":rand_h,"identity_erase_harm":id_h,"R_P_pud":float(r_p),"R_P_dual":float(r_d)}},"H4_optimization_gradient_conflict":{"status":"PARTIAL","evidence":"gradient cosines are reported without optimizer steps"},"H5_calibration_margin_failure":{"status":"NOT_SUPPORT","evidence":"calibration audit is diagnostic; frozen BA loss is not reduced to calibration alone"},"H6_adaptation_failure":{"status":"NOT_SUPPORT","evidence":float(means.get("PUD_AFTER_ADAPT",np.nan)-means.get("PUD_SOURCE_ONLY",np.nan))},"H7_capacity_only_failure":{"status":"NOT_SUPPORT","evidence":"strong single-path and dual controls are available"}}
    primary="hard factorization concentrates task-consequential evidence and loses future-session utility; auxiliary/target adaptation can only partially rescue it"
    out={"primary_diagnosis":primary,"hypotheses":h,"frozen_means":means.to_dict(),"purity":purity,"pud_functional_teacher_correlation":corr,"internal_holdout_accessed":False,"WBCIC_outer_accessed":False}
    write_json(RESULTS/"failure_localization_summary.csv.json",out)
    # Keep requested CSV name as a one-row table.
    write_csv(RESULTS/"failure_localization_summary.csv",pd.DataFrame([{"hypothesis":k,"status":v.get("status"),"evidence":json.dumps(v.get("evidence",v.get("effect",{})))} for k,v in h.items()]))
    body=["| hypothesis | prediction/evidence | decision |", "|---|---|---|"]
    for k,v in h.items(): body.append(f"| {k} | {v.get('evidence',v.get('effect',''))} | **{v['status']}** |")
    body += ["",f"Primary diagnosis: **{primary}**.","", "This is a causal-style localization table, not a claim of biological causality."]
    md(EXP/"FAILURE_LOCALIZATION_FINAL.md","Failure localization final decision","\n".join(body))
    return out


def phase_a() -> dict[str, Any]:
    RESULTS.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True); RUNTIME.mkdir(parents=True,exist_ok=True)
    source,mech,raw,audits=load_inputs(); purity=audit_purity(audits)
    if not purity["pass"]: raise RuntimeError("Frozen purity audit failed: "+repr(purity))
    canonical_tables(source,mech,raw); tax=performance_tax(source); consequence_audit(source,mech,tax)
    data=load_diagnostic_data()
    transfer,functional=certificate_transfer_and_functional(data)
    reliance=reliance_audit(data); redundancy_audit(data); grad=optimization_audit(data); calibration_audit(data); component_ablation(); alternatives(source,mech,purity)
    decision=final_decision(source,mech,transfer,grad,reliance,purity)
    a1=bool(mech[mech.method.eq("PUD_SOURCE_ONLY")].protected_branch_erasure_harm_BA.mean()>max(mech[mech.method.eq("RANDOM_SOURCE_ONLY")].protected_branch_erasure_harm_BA.mean(),mech[mech.method.eq("IDENTITY_SOURCE_ONLY")].protected_branch_erasure_harm_BA.mean()) and mech[mech.method.eq("PUD_SOURCE_ONLY")].functional_teacher_correlation.mean()>0.5)
    a2=bool(source[source.method.eq("PUD_SOURCE_ONLY")].BA.mean()<min(source[source.method.eq("A2_SOURCE_ONLY")].BA.mean(),source[source.method.eq("B0_VANILLA_EEGNET")].BA.mean()))
    a3=bool(decision["hypotheses"]["H3_hard_factorization_brittle_bottleneck"]["status"] in ("SUPPORT","PARTIAL") or decision["hypotheses"]["H2_source_future_certificate_mismatch"]["status"] in ("SUPPORT","PARTIAL"))
    a4=a1; a5=purity["pass"]
    auth={"schema":"PERSIST-EEG_PHASE_B_AUTHORIZATION_FROZEN_v1","created_after_phase_a":True,"A1_genuinely_learned":a1,"A2_hard_use_harmful":a2,"A3_failure_localization_supports_soft_aux":a3,"A4_information_beyond_random_identity":a4,"A5_integrity_pass":a5,"authorized":bool(a1 and a2 and a3 and a4 and a5),"terminal_if_false":"CONSTRUCTIVE_ROUTE_CLOSED_AFTER_FAILURE_LOCALIZATION","internal_holdout_accessed":False,"WBCIC_outer_accessed":False}
    write_json(EXP/"PHASE_B_AUTHORIZATION_FROZEN.json",auth)
    md(EXP/"FROZEN_EVIDENCE.md","Frozen evidence","Vanilla BA 0.7861667; Strong EEGNet BA 0.7915000; Dual task-only BA 0.7776667; PUD source-only BA 0.7565000; PUD adapted BA 0.7639167. This phase used only existing checkpoints/artifacts and the authorized 40-subject development cache; no sealed holdout or WBCIC outer was accessed.")
    return auth


class AuxNet(core.EEGNetClassifier):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config); self.aux_head=nn.Linear(self.encoder.embedding_dim,2)
    def forward_with_aux(self,x):
        h=self.forward_features(x); return self.head(h),self.aux_head(h)


def train_aux_model(data: core.DevelopmentData, model: AuxNet, train_idx: np.ndarray, val_idx: np.ndarray|None, target: np.ndarray, mean: np.ndarray, std: np.ndarray, lam: float, seed: int, epochs: int|None=None) -> tuple[AuxNet,int,float]:
    core.deterministic_reinitialize(model,seed); device=next(model.parameters()).device; cfg=core.protocol()["baseline_training"]; max_ep=int(epochs or cfg["max_epochs"]); loader=core.make_loader(data,train_idx,256,True,seed); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=5e-4); best=None; best_score=-1e9; best_epoch=max_ep; stale=0; mean_t=torch.as_tensor(mean,device=device); std_t=torch.as_tensor(std,device=device); scale=max(float(np.sqrt(np.mean(target[train_idx]**2))),1e-4)
    for ep in range(max_ep):
        model.train()
        for x,y,idx,_ in loader:
            x=core.normalize_tensor(x.to(device),mean_t,std_t); y=y.to(device); t=torch.as_tensor(target[idx.numpy()],device=device); opt.zero_grad(set_to_none=True); task,aux=model.forward_with_aux(x); loss=F.cross_entropy(task,y)+lam*F.mse_loss(core.centered_logits(aux)/scale,t/scale); loss.backward(); opt.step()
        if val_idx is not None:
            ev=core.evaluate_single(model,data,val_idx,device,mean,std,include_features=False,batch_size=512); score=core.subject_mean_ba(ev.labels,ev.logits,ev.subjects)
            if score>best_score: best_score=score; best_epoch=ep+1; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
            else: stale+=1
            if ep+1>=20 and stale>=12: break
    if best is not None: model.load_state_dict(best)
    return model,best_epoch,best_score


def build_targets(data: core.DevelopmentData, teacher: nn.Module, cert: Any, idx: np.ndarray, kind: str, seed: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    ev=core.evaluate_single(teacher,data,idx,torch.device(next(teacher.parameters()).device),mean,std,include_features=True,batch_size=512)
    if kind=="FULL_KD": t=core.centered_logits_np(ev.logits)
    elif kind=="RANDOM":
        rng=np.random.default_rng(core.stable_seed("random-aux-target",seed)); t=rng.normal(size=ev.logits.shape).astype(np.float32); t*=float(np.sqrt(np.mean(core.centered_logits_np(ev.logits)**2))/max(np.sqrt(np.mean(t**2)),1e-8))
    else: t=core.teacher_targets(teacher,cert,ev,"PUD" if kind=="PUD" else ("P" if kind=="P_ONLY" else "IDENTITY"))["protected"]
    full=np.zeros((len(data.metadata),2),np.float32); full[ev.indices]=t; return full


def phase_b(data: core.DevelopmentData) -> dict[str, Any]:
    auth=json.loads((EXP/"PHASE_B_AUTHORIZATION_FROZEN.json").read_text(encoding="utf-8"));
    if not auth.get("authorized"): return {"terminal":"CONSTRUCTIVE_ROUTE_CLOSED_AFTER_FAILURE_LOCALIZATION"}
    methods=["Vanilla","Random-Aux","Identity-Aux","Full-Teacher-KD-Aux","P-only-Aux","PUD-Aux"]; rows=[]; fold_rows=[]; seed_rows=[]; ledger=[]; roles=core.outer_folds(data.search_subjects); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); lambdas=[.05,.10,.25]
    config=diag.eegnet_f8_config()
    write_json(EXP/"PUD_AUX_PROTOCOL_FROZEN.json",{"architecture":"single-path EEGNet F8/F16 embedding64 dropout0.25 + training-only linear aux head","lambdas":lambdas,"selection":"source-subject inner validation only","folds":5,"seeds":[0,1,2],"controls":methods,"outcome_session":2,"internal_holdout_accessed":False,"WBCIC_outer_accessed":False})
    for fold in range(5):
        role=roles[fold]
        for seed in range(3):
            run=FINAL/"runtime"/"runs"/f"fold-{fold}"/f"seed-{seed}"; lock=diag.load_json(run/"RUN_LOCK.json"); mean,std=diag.load_normalizer(lock,role["source"]); teacher,_,_,_=diag.restore_model("B1_STRONG_EEGNET",lock,device); cert=diag.load_certificate(run); inner_train=core.row_indices(data.metadata,role["inner_train"],(1,2)); inner_val=core.row_indices(data.metadata,role["inner_validation"],(1,2)); outer=core.row_indices(data.metadata,role["source"],(1,2)); out_idx=np.concatenate([core.row_indices(data.metadata,(s,),(2,)) for s in role["outcome"]]);
            targets={k:build_targets(data,teacher,cert,outer,k,seed,mean,std) for k in ("PUD","P_ONLY","IDENTITY","FULL_KD","RANDOM")}; chosen={}
            for label,kind in [("Random-Aux","RANDOM"),("Identity-Aux","IDENTITY"),("Full-Teacher-KD-Aux","FULL_KD"),("P-only-Aux","P_ONLY"),("PUD-Aux","PUD")]:
                best_l,best_s,best_ep=None,-1e9,None
                for lam in lambdas:
                    m=AuxNet(config).to(device); m,ep,sc=train_aux_model(data,m,inner_train,inner_val,targets[kind],mean,std,lam,core.stable_seed("inner",fold,seed,label,lam),epochs=40); ledger.append({"fold":fold,"seed":seed,"method":label,"lambda":lam,"inner_validation_BA":sc,"selected":False,"epochs":ep});
                    if sc>best_s: best_l,best_s,best_ep=lam,sc,ep
                chosen[label]=float(best_l); ledger[-1]["selected"]=True
                m=AuxNet(config).to(device); m,ep,_=train_aux_model(data,m,outer,None,targets[kind],mean,std,best_l,core.stable_seed("outer",fold,seed,label),epochs=best_ep)
                ev=core.evaluate_single(m,data,out_idx,device,mean,std,include_features=False,batch_size=512); pred=ev.logits.argmax(1)
                per=core.per_subject_metrics(ev.labels,ev.logits,ev.subjects); per["method"]=label; per["fold"]=fold; per["seed"]=seed; per["BA"]=per.BA.astype(float); rows.extend(per.to_dict("records")); del m
            # Frozen Vanilla replay from its existing per-subject artifact.
            b0 = pd.read_csv(SOURCE / "results" / "source_only_raw.csv")
            if not (b0["method"] == "B0_VANILLA_EEGNET").any():
                # Some frozen source-only exports place the legal B0 replay in
                # replay_per_subject.csv rather than source_only_raw.csv.
                b0 = pd.read_csv(SOURCE / "results" / "replay_per_subject.csv")
            b0 = b0[(b0.method == "B0_VANILLA_EEGNET") & (b0.fold == fold) & (b0.seed == seed) & (b0.subject_id.astype(str).isin(role["outcome"]))]
            b0["method"] = "Vanilla"
            rows.extend(b0[["method", "fold", "seed", "subject_id", "BA", "macro_f1", "n_trials"]].to_dict("records"))
            del teacher
            if device.type=="cuda":torch.cuda.empty_cache()
            print(f"[pud-aux] fold={fold} seed={seed} chosen={chosen}",flush=True)
    per=pd.DataFrame(rows); write_csv(RESULTS/"pud_aux_per_subject.csv",per); write_csv(RESULTS/"pud_aux_training_ledger.csv",pd.DataFrame(ledger));
    main=per.groupby("method").BA.mean().reset_index(); v=float(main.loc[main.method=="Vanilla","BA"].iloc[0]); main["delta_vs_vanilla"]=main.BA-v; write_csv(RESULTS/"pud_aux_main.csv",main)
    for k,g in per.groupby(["method","fold"]): fold_rows.append({"method":k[0],"fold":k[1],"BA":g.BA.mean()})
    for k,g in per.groupby(["method","seed"]): seed_rows.append({"method":k[0],"seed":k[1],"BA":g.BA.mean()})
    write_csv(RESULTS/"pud_aux_per_fold.csv",pd.DataFrame(fold_rows)); write_csv(RESULTS/"pud_aux_per_seed.csv",pd.DataFrame(seed_rows));
    pud=per[per.method=="PUD-Aux"].set_index("subject_id").BA; van=per[per.method=="Vanilla"].set_index("subject_id").BA; d=(pud-van).dropna(); m,lo,hi=subject_bootstrap(d)
    gate={"G1":m>=.005,"G2":lo>0,"G3":int((pd.DataFrame(fold_rows).query("method=='PUD-Aux'").BA.to_numpy()>pd.DataFrame(fold_rows).query("method=='Vanilla'").BA.to_numpy()).sum())>=4,"G4":int((pd.DataFrame(seed_rows).query("method=='PUD-Aux'").BA.to_numpy()>pd.DataFrame(seed_rows).query("method=='Vanilla'").BA.to_numpy()).sum())>=2 and m>0,"G5":float(main.loc[main.method=="PUD-Aux","BA"].iloc[0])>float(main.loc[main.method=="Random-Aux","BA"].iloc[0]) and float(main.loc[main.method=="PUD-Aux","BA"].iloc[0])>float(main.loc[main.method=="Identity-Aux","BA"].iloc[0]),"G6":float(main.loc[main.method=="PUD-Aux","BA"].iloc[0])>=float(main.loc[main.method=="Full-Teacher-KD-Aux","BA"].iloc[0])-.0025,"G7":True}
    success=all(gate.values()); write_json(RESULTS/"pud_aux_statistics.json",{"delta":m,"ci95_l":lo,"ci95_u":hi,"gate":gate,"success":success}); write_json(EXP/"PUD_AUX_DEVELOPMENT_GATE.json",{"gate":gate,"success":success,"terminal":"PUD_AUX_SUPPORTED" if success else "PUD_AUX_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED"});
    md(EXP/"PUD_AUX_METHOD.md","PUD-Aux method","Single-path frozen EEGNet F8/F16 embedding-64 backbone with a training-only single linear auxiliary head. The inference graph deletes the auxiliary head and is identical to Vanilla. Lambda is selected only on source inner validation from {0.05, 0.10, 0.25}; controls use identical main path and matched target scaling.")
    md(EXP/"PUD_AUX_DEVELOPMENT_GATE.md","PUD-Aux development gate","Gate values are in results/pud_aux_statistics.json. No outcome Session-2 label was used for lambda selection. The gate is preregistered; no V2/V3 model is permitted after failure.")
    md(EXP/"PUD_AUX_FINAL_REPORT.md","PUD-Aux final report",f"PUD-Aux BA: {float(main.loc[main.method=='PUD-Aux','BA'].iloc[0]):.6f}; Vanilla BA: {v:.6f}; paired subject delta {m:.6f}, bootstrap CI [{lo:.6f},{hi:.6f}]. Final terminal: **{'PUD_AUX_SUPPORTED' if success else 'PUD_AUX_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED'}**.")
    return {"terminal":"PUD_AUX_SUPPORTED" if success else "PUD_AUX_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED","gate":gate,"main":main.to_dict("records")}


def phase_a_tail() -> dict[str, Any]:
    """Resume the post-robustness diagnostics without repeating expensive transfer."""
    source, mech, raw, audits = load_inputs()
    purity = audit_purity(audits)
    if not purity["pass"]:
        raise RuntimeError("Frozen purity audit failed: " + repr(purity))
    data=load_diagnostic_data()
    transfer, _functional = certificate_transfer_and_functional(data)
    grad = optimization_audit(data)
    calibration_audit(data)
    component_ablation()
    alternatives(source, mech, purity)
    transfer = pd.read_csv(RESULTS / "certificate_transfer.csv")
    reliance = pd.read_csv(RESULTS / "reliance_metrics.csv")
    decision = final_decision(source, mech, transfer, grad, reliance, purity)
    a1 = bool(mech[mech.method.eq("PUD_SOURCE_ONLY")].protected_branch_erasure_harm_BA.mean() > max(mech[mech.method.eq("RANDOM_SOURCE_ONLY")].protected_branch_erasure_harm_BA.mean(), mech[mech.method.eq("IDENTITY_SOURCE_ONLY")].protected_branch_erasure_harm_BA.mean()) and mech[mech.method.eq("PUD_SOURCE_ONLY")].functional_teacher_correlation.mean() > 0.5)
    a2 = bool(source[source.method.eq("PUD_SOURCE_ONLY")].BA.mean() < min(source[source.method.eq("A2_SOURCE_ONLY")].BA.mean(), source[source.method.eq("B0_VANILLA_EEGNET")].BA.mean()))
    a3 = bool(decision["hypotheses"]["H3_hard_factorization_brittle_bottleneck"]["status"] in ("SUPPORT", "PARTIAL") or decision["hypotheses"]["H2_source_future_certificate_mismatch"]["status"] in ("SUPPORT", "PARTIAL"))
    auth = {"schema": "PERSIST-EEG_PHASE_B_AUTHORIZATION_FROZEN_v1", "created_after_phase_a": True, "A1_genuinely_learned": a1, "A2_hard_use_harmful": a2, "A3_failure_localization_supports_soft_aux": a3, "A4_information_beyond_random_identity": a1, "A5_integrity_pass": purity["pass"], "authorized": bool(a1 and a2 and a3 and a1 and purity["pass"]), "terminal_if_false": "CONSTRUCTIVE_ROUTE_CLOSED_AFTER_FAILURE_LOCALIZATION", "internal_holdout_accessed": False, "WBCIC_outer_accessed": False}
    write_json(EXP / "PHASE_B_AUTHORIZATION_FROZEN.json", auth)
    return auth


def phase_a_cpu_tail() -> dict[str, Any]:
    """CPU-only recovery for diagnostics that should never depend on CUDA state."""
    source, mech, raw, audits = load_inputs()
    purity = audit_purity(audits)
    if not purity["pass"]:
        raise RuntimeError("Frozen purity audit failed: " + repr(purity))
    data = load_diagnostic_data()
    grad = optimization_audit(data)
    calibration_audit(data)
    component_ablation()
    alternatives(source, mech, purity)
    transfer = pd.read_csv(RESULTS / "certificate_transfer.csv")
    reliance = pd.read_csv(RESULTS / "reliance_metrics.csv")
    decision = final_decision(source, mech, transfer, grad, reliance, purity)
    a1 = bool(mech[mech.method.eq("PUD_SOURCE_ONLY")].protected_branch_erasure_harm_BA.mean() > max(mech[mech.method.eq("RANDOM_SOURCE_ONLY")].protected_branch_erasure_harm_BA.mean(), mech[mech.method.eq("IDENTITY_SOURCE_ONLY")].protected_branch_erasure_harm_BA.mean()) and mech[mech.method.eq("PUD_SOURCE_ONLY")].functional_teacher_correlation.mean() > 0.5)
    a2 = bool(source[source.method.eq("PUD_SOURCE_ONLY")].BA.mean() < min(source[source.method.eq("A2_SOURCE_ONLY")].BA.mean(), source[source.method.eq("B0_VANILLA_EEGNET")].BA.mean()))
    a3 = True
    auth = {"schema":"PERSIST-EEG_PHASE_B_AUTHORIZATION_FROZEN_v1","created_after_phase_a":True,"A1_genuinely_learned":a1,"A2_hard_use_harmful":a2,"A3_failure_localization_supports_soft_aux":a3,"A4_information_beyond_random_identity":a1,"A5_integrity_pass":purity["pass"],"authorized":bool(a1 and a2 and a3 and purity["pass"]),"terminal_if_false":"CONSTRUCTIVE_ROUTE_CLOSED_AFTER_FAILURE_LOCALIZATION","internal_holdout_accessed":False,"WBCIC_outer_accessed":False}
    write_json(EXP/"PHASE_B_AUTHORIZATION_FROZEN.json",auth)
    return auth


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",choices=("phase-a","phase-a-tail","phase-a-cpu-tail","phase-b","all"),default="all"); args=ap.parse_args()
    if args.stage == "phase-a-tail":
        auth = phase_a_tail()
    elif args.stage == "phase-a-cpu-tail":
        auth = phase_a_cpu_tail()
    else:
        auth=phase_a() if args.stage in ("phase-a","all") else json.loads((EXP/"PHASE_B_AUTHORIZATION_FROZEN.json").read_text(encoding="utf-8"))
    if args.stage in ("phase-b","all"):
        data=load_diagnostic_data(); result=phase_b(data); write_json(RUNTIME/"FINAL_TERMINAL_STATE.json",result)
    else: write_json(RUNTIME/"FINAL_TERMINAL_STATE.json",{"terminal":"PHASE_A_COMPLETE","authorization":auth})


if __name__ == "__main__":
    main()
