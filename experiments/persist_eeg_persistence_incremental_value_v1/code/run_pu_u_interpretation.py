"""Frozen PU-vs-U-only explanatory analysis.

This is deliberately downstream of Experiment 1: it neither trains a network
nor changes a selector.  Per-cell files are resumable and contain only compact
subject/block statistics; checkpoints and EEG/cache tensors remain external.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import run_cell


MODELS = ("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer", "EEGConformer", "FBCNet")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
OUT = run_cell.EXP / "outputs"
CELL_ROOT = run_cell.RUNTIME / "pu_u_interpretation"
EXECUTION_MODE = os.environ.get("PERSIST_EXECUTION_MODE", "FROZEN_REPLAY")
CPU_EXPLORATORY = EXECUTION_MODE == "CPU_EXPLORATORY_NON_EQUIVALENT"


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32 - 1)


def path_for(model: str, task: str, fold: int, seed: int) -> Path:
    return CELL_ROOT / model.lower() / task.lower() / f"fold{fold}_seed{seed}.json"


def write_csv(name: str, rows: list[dict]) -> None:
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def mean_subject(values: dict[str, float]) -> float:
    return float(np.mean(list(values.values()))) if values else float("nan")


def score_by_session(q_source: np.ndarray, source_y: np.ndarray, representations: dict[str, np.ndarray],
                     evaluation: dict[str, dict], dims: list[int], classes: int) -> dict[str, dict[str, float]]:
    if not dims:
        # ``ridge_scores`` has the closed-form no-coordinate solution
        # ``score = mean(one_hot(source_y))``.  Calling it with an [N,0]
        # array instead emits empty-slice warnings; spell out the exact same
        # argmax decision so rank-complete random removals remain defined.
        prior_class = int(np.eye(classes, dtype=np.float64)[source_y].mean(0).argmax())
        return {name: run_cell.ba_by_subject(np.full(len(value["y"]), prior_class, dtype=np.int64),
                                             value["y"], value["subjects"].astype(str))
                for name, value in evaluation.items()}
    result = {}
    for name, q_eval in representations.items():
        value = evaluation[name]
        prediction = run_cell.pswa.ridge_scores(q_source[:, dims], source_y, q_eval[:, dims], classes).argmax(1)
        result[name] = run_cell.ba_by_subject(prediction, value["y"], value["subjects"].astype(str))
    return result


def endpoints(scores: dict[str, dict[str, float]], future_name: str) -> tuple[dict[str, float], dict[str, float]]:
    future = scores[future_name]
    worst = {subject: min(by_subject[subject] for by_subject in scores.values()) for subject in future}
    return future, worst


def cell(model: str, task: str, fold: int, seed: int) -> None:
    target = path_for(model, task, fold, seed)
    if target.is_file():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old.get("identity") == [model, task, fold, seed]:
            print("INTERPRETATION_CACHED", model, task, fold, seed, flush=True); return
        raise RuntimeError("stale interpretation cache identity")
    source_path = run_cell.result_path(model, task, fold, seed)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("identity") != [model, task, fold, seed]:
        raise RuntimeError("Experiment 1 cell identity mismatch")
    base = {"identity": [model, task, fold, seed], "source_cell": str(source_path),
            "source_status": source["status"], "training_performed": False,
            "selector_changed": False, "heldout_used_for_selection": False,
            "execution_mode": EXECUTION_MODE,
            "numerical_replay_equivalence": not CPU_EXPLORATORY}
    if source["status"] != "ESTIMABLE":
        target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(base, indent=2) + "\n"); return
    run_cell.gate()
    audit = run_cell.audit_row(model, task, fold, seed)
    checkpoint = Path(audit.get("local_checkpoint_path", audit["checkpoint_path"]))
    if run_cell.digest(checkpoint) != audit["checkpoint_sha256"] or source["checkpoint_sha256"] != audit["checkpoint_sha256"]:
        raise RuntimeError("checkpoint provenance changed")
    record = json.loads((checkpoint.parent / "record.json").read_text(encoding="utf-8"))
    data = run_cell.pswa.load_arrays(task, fold)
    if data["normalizer"]["mean_std_sha256"] != audit["normalizer_sha256"]:
        raise RuntimeError("normalizer provenance changed")
    capped = run_cell.pswa.cap_data(data, task, fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, head = run_cell.build_model(model, record, checkpoint, device)
    hs = run_cell.representation(net, head, capped["source_x"], model, device)
    hf = run_cell.representation(net, head, capped["future_x"], model, device)
    h = np.concatenate([hs, hf]); y = np.concatenate([capped["source_y"], capped["future_y"]])
    owner = np.concatenate([capped["source_subjects"], capped["future_subjects"]])
    sessions = np.concatenate([np.full(len(hs), data["source_session"]), np.full(len(hf), data["future_session"])]).astype(np.int64)
    spec = run_cell.peeh.spectrum(h, y, owner, sessions, task, model, fold); spec["classes"] = data["classes"]
    spectrum_matches_frozen = (int(spec["rank"]) == int(source["active_rank"]) and spec["blocks"] == source["blocks"])
    if not CPU_EXPLORATORY and not spectrum_matches_frozen:
        raise RuntimeError("frozen spectrum differs from Experiment 1")
    assignments = source["utility_evidence"]
    stored = {name: tuple(map(int, source["selector_coordinates"][name])) for name in ("PU", "U_only", "P_only")}
    if not CPU_EXPLORATORY:
        selected = run_cell.selectors.choose(spec, assignments)
        if any(tuple(selected[name].coordinates) != stored[name] for name in stored):
            raise RuntimeError("selector reconstruction mismatch")
    q_source = run_cell.peeh.canonical(hs, spec).astype(np.float32)
    representations, evaluation = {}, capped["evaluation"]
    for name, value in evaluation.items():
        representations[name] = run_cell.peeh.canonical(run_cell.representation(net, head, value["x"], model, device), spec).astype(np.float32)
    future_name = f"S{data['future_session']}"
    all_selector = {}
    for name in ("PU", "U_only"):
        future, worst = endpoints(score_by_session(q_source, capped["source_y"], representations, evaluation,
                                                   list(stored[name]), data["classes"]), future_name)
        all_selector[name] = {"rank": len(stored[name]), "future": future, "worst": worst}
    u_blocks = list(map(int, source["selector_blocks"]["U_only"]))
    persistent_blocks = [b for b in u_blocks if bool(source["support"][b]["persistence_supported"])]
    nonpersistent_blocks = [b for b in u_blocks if b not in persistent_blocks]
    blocks = [list(map(int, b)) for b in source["blocks"]]
    p_dims = sorted(i for b in persistent_blocks for i in blocks[b]); n_dims = sorted(i for b in nonpersistent_blocks for i in blocks[b])
    u_dims = list(stored["U_only"])
    n_only = {"future": {}, "worst": {}}
    if n_dims:
        n_only["future"], n_only["worst"] = endpoints(score_by_session(q_source, capped["source_y"], representations, evaluation, n_dims, data["classes"]), future_name)
    if p_dims:
        minus_future, minus_worst = endpoints(score_by_session(q_source, capped["source_y"], representations, evaluation, p_dims, data["classes"]), future_name)
    else:
        minus_future = {s: 0.0 for s in all_selector["U_only"]["future"]}; minus_worst = dict(minus_future)
    random_harms_f, random_harms_w = defaultdict(list), defaultdict(list)
    if n_dims:
        for draw in range(100):
            remove = set(np.random.default_rng(stable_seed("u-nonpersistent-removal", model, task, fold, seed, draw)).choice(u_dims, len(n_dims), replace=False).tolist())
            kept = [d for d in u_dims if d not in remove]
            rf, rw = endpoints(score_by_session(q_source, capped["source_y"], representations, evaluation, kept, data["classes"]), future_name)
            for subject, value in all_selector["U_only"]["future"].items():
                random_harms_f[subject].append(value - rf[subject]); random_harms_w[subject].append(all_selector["U_only"]["worst"][subject] - rw[subject])
    nonpersistent = []
    for subject, u_future in all_selector["U_only"]["future"].items():
        hf = u_future - minus_future[subject] if n_dims else 0.0; hw = all_selector["U_only"]["worst"][subject] - minus_worst[subject] if n_dims else 0.0
        nonpersistent.append({"subject_id": subject, "rank_N": len(n_dims), "persistent_rank": len(p_dims),
                              "N_future_BA": n_only["future"].get(subject), "N_worst_BA": n_only["worst"].get(subject),
                              "erasure_harm_future": hf, "erasure_harm_worst": hw,
                              "random_removal_harm_future": float(np.mean(random_harms_f[subject])) if n_dims else 0.0,
                              "random_removal_harm_worst": float(np.mean(random_harms_w[subject])) if n_dims else 0.0,
                              "excess_future": hf - (float(np.mean(random_harms_f[subject])) if n_dims else 0.0),
                              "excess_worst": hw - (float(np.mean(random_harms_w[subject])) if n_dims else 0.0)})
    blocks_out = []
    for block_id, block_dims in enumerate(blocks):
        score_u = min(float(assignments[block_id]["absolute_CI_low"]), float(assignments[block_id]["excess_CI_low"]))
        if score_u <= 0: continue
        bf, bw = endpoints(score_by_session(q_source, capped["source_y"], representations, evaluation, block_dims, data["classes"]), future_name)
        blocks_out.append({"block_id": block_id, "block_rank": len(block_dims), "score_U": score_u,
                           "persistence_margin": float(source["support"][block_id]["rho"]) - float(source["support"][block_id]["null_p95"]),
                           "persistence_pass": bool(source["support"][block_id]["persistence_supported"]),
                           "future": bf, "worst": bw})
    result = {**base, "spectrum_matches_frozen": spectrum_matches_frozen,
              "rank": {name: len(stored[name]) for name in stored}, "selector_blocks": source["selector_blocks"],
              "selector_coordinates": source["selector_coordinates"], "selector_absolute": all_selector,
              "u_composition": {"persistent_blocks": persistent_blocks, "nonpersistent_blocks": nonpersistent_blocks,
                                "persistent_rank": len(p_dims), "nonpersistent_rank": len(n_dims)},
              "nonpersistent": nonpersistent, "utility_positive_blocks": blocks_out}
    target.parent.mkdir(parents=True, exist_ok=True); temporary = target.with_suffix(".json.part")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); temporary.replace(target)
    print("INTERPRETATION_COMPLETE", model, task, fold, seed, flush=True)


def bootstrap(values: dict[str, list[float]], *parts: object) -> tuple[float, float, float, int]:
    subjects = sorted(values)
    if not subjects: return float("nan"), float("nan"), float("nan"), 0
    a = np.asarray([np.mean(values[s]) for s in subjects], dtype=float); rng = np.random.default_rng(stable_seed("pu-u-bootstrap", *parts))
    draws = a[rng.integers(0, len(a), size=(20_000, len(a)))].mean(1)
    return float(a.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), len(a)


def aggregate(selected_models: tuple[str, ...] = MODELS) -> None:
    source_cells, interpretation = [], []
    for model in selected_models:
        for task in TASKS:
            for fold in range(5):
                for seed in range(3):
                    original = run_cell.result_path(model, task, fold, seed)
                    derived = path_for(model, task, fold, seed)
                    if not original.is_file() or not derived.is_file(): raise RuntimeError(f"incomplete cell: {(model, task, fold, seed)}")
                    source_cells.append(json.loads(original.read_text(encoding="utf-8"))); interpretation.append(json.loads(derived.read_text(encoding="utf-8")))
    rank_rows=[]; overlap_rows=[]; composition_rows=[]; absolute_rows=[]; nonp_rows=[]; block_rows=[]
    for src, der in zip(source_cells, interpretation):
        model, task, fold, seed = src["identity"]; base={"model":model,"task":task,"fold":fold,"seed":seed,"status":src["status"]}
        if src["status"] != "ESTIMABLE":
            rank_rows.append({**base,"active_rank":src.get("active_rank",""),"rank_PU":0,"rank_Uonly":0,"rank_Ponly":0,"rank_matched":True}); continue
        ranks=der["rank"]; pu=set(map(int,der["selector_coordinates"]["PU"])); u=set(map(int,der["selector_coordinates"]["U_only"])); k=ranks["PU"]
        if not(k==ranks["U_only"]==ranks["P_only"]): raise RuntimeError("rank audit failed")
        shared=sorted(pu & u); rank_rows.append({**base,"active_rank":src["active_rank"],"rank_PU":k,"rank_Uonly":ranks["U_only"],"rank_Ponly":ranks["P_only"],"rank_matched":True})
        overlap_rows.append({**base,"rank":k,"intersection_rank":len(shared),"overlap_fraction":len(shared)/k,"jaccard":len(shared)/(2*k-len(shared)),"exact_same_selection":pu==u,"PU_block_ids":";".join(map(str,der["selector_blocks"]["PU"])),"Uonly_block_ids":";".join(map(str,der["selector_blocks"]["U_only"])),"shared_block_ids":";".join(map(str,sorted(set(der["selector_blocks"]["PU"])&set(der["selector_blocks"]["U_only"]))) )})
        comp=der["u_composition"]; composition_rows.append({**base,"rank_Uonly":k,"persistent_rank_in_U":comp["persistent_rank"],"nonpersistent_rank_in_U":comp["nonpersistent_rank"],"persistent_fraction_in_U":comp["persistent_rank"]/k})
        for selector in ("PU","U_only"):
            for subject, future in der["selector_absolute"][selector]["future"].items(): absolute_rows.append({**base,"selector":selector,"subject_id":subject,"future_BA":future,"worst_BA":der["selector_absolute"][selector]["worst"][subject]})
        for row in der["nonpersistent"]: nonp_rows.append({**base,**row})
        for block in der["utility_positive_blocks"]:
            for subject, future in block["future"].items(): block_rows.append({**base,"block_id":block["block_id"],"block_rank":block["block_rank"],"score_U":block["score_U"],"persistence_margin":block["persistence_margin"],"persistence_pass":block["persistence_pass"],"subject_id":subject,"future_BA":future,"worst_BA":block["worst"][subject]})
    write_csv("PU_U_RANK_AUDIT.csv",rank_rows);write_csv("PU_U_SELECTION_OVERLAP.csv",overlap_rows);write_csv("UONLY_PERSISTENCE_COMPOSITION.csv",composition_rows);write_csv("PU_U_ABSOLUTE_UTILITY.csv",absolute_rows);write_csv("UONLY_NONPERSISTENT_UTILITY.csv",nonp_rows);write_csv("UONLY_NONPERSISTENT_ERASURE.csv",nonp_rows);write_csv("UTILITY_POSITIVE_BLOCKS.csv",block_rows)
    paired=[]
    for (model,task), rows in _groups(absolute_rows,lambda r:(r["model"],r["task"])).items():
        by=defaultdict(dict)
        for r in rows: by[r["subject_id"]].setdefault(r["selector"],[]).append(r)
        for subject, values in by.items():
            if "PU" in values and "U_only" in values:
                paired.append({"model":model,"task":task,"subject_id":subject,"delta_absolute_future":float(np.mean([r["future_BA"] for r in values["U_only"]]))-float(np.mean([r["future_BA"] for r in values["PU"]])),"delta_absolute_worst":float(np.mean([r["worst_BA"] for r in values["U_only"]]))-float(np.mean([r["worst_BA"] for r in values["PU"]]))})
    write_csv("PU_U_ABSOLUTE_PAIRED_EFFECTS.csv",paired)
    pairs=[]
    for (model,task,fold,seed), rows in _groups(block_rows,lambda r:(r["model"],r["task"],r["fold"],r["seed"])).items():
        p=[r for r in rows if r["persistence_pass"]]; n=[r for r in rows if not r["persistence_pass"]]
        for pr in sorted({r["block_id"] for r in p}):
            proto=next(r for r in p if r["block_id"]==pr); candidates=[r for r in n if r["block_rank"]==proto["block_rank"]]
            if not candidates: continue
            chosen=min(candidates,key=lambda r:(abs(r["score_U"]-proto["score_U"]),r["block_id"]))
            for subject in sorted({r["subject_id"] for r in rows}):
                pp=next(r for r in p if r["block_id"]==pr and r["subject_id"]==subject); nn=next(r for r in n if r["block_id"]==chosen["block_id"] and r["subject_id"]==subject)
                pairs.append({"model":model,"task":task,"fold":fold,"seed":seed,"subject_id":subject,"persistent_block":pr,"nonpersistent_block":chosen["block_id"],"block_rank":proto["block_rank"],"score_U_persistent":proto["score_U"],"score_U_nonpersistent":chosen["score_U"],"Delta_future":pp["future_BA"]-nn["future_BA"],"Delta_worst":pp["worst_BA"]-nn["worst_BA"]})
    write_csv("UTILITY_MATCHED_BLOCK_PAIRS.csv",pairs)
    subject_effects=[]; summaries=[]
    for (model,task), rows in _groups(pairs,lambda r:(r["model"],r["task"])).items():
        values_f=defaultdict(list);values_w=defaultdict(list)
        for r in rows: values_f[r["subject_id"]].append(r["Delta_future"]);values_w[r["subject_id"]].append(r["Delta_worst"])
        for subject in values_f: subject_effects.append({"model":model,"task":task,"subject_id":subject,"Delta_future":np.mean(values_f[subject]),"Delta_worst":np.mean(values_w[subject])})
        mf,lf,hf,nf=bootstrap(values_f,model,task,"future");mw,lw,hw,nw=bootstrap(values_w,model,task,"worst");summaries.append({"model":model,"task":task,"matched_pairs":len({(r['fold'],r['seed'],r['persistent_block']) for r in rows}),"subjects":nf,"Delta_future":mf,"future_CI_low":lf,"future_CI_high":hf,"Delta_worst":mw,"worst_CI_low":lw,"worst_CI_high":hw})
    write_csv("UTILITY_MATCHED_SUBJECT_EFFECTS.csv",subject_effects);write_csv("UTILITY_MATCHED_TASK_SUMMARY.csv",summaries)
    # Descriptive only: OLS coefficient; blocks are never treated as biological independent samples.
    regression=[]
    for (model,task), rows in _groups(block_rows,lambda r:(r["model"],r["task"])).items():
        grouped=defaultdict(list)
        for r in rows: grouped[(r["fold"],r["seed"],r["block_id"])].append(r)
        table=[]
        for rs in grouped.values(): table.append((np.mean([r["future_BA"] for r in rs]),rs[0]["score_U"],rs[0]["persistence_margin"],rs[0]["block_rank"]))
        if len(table)>=4:
            y=np.asarray([r[0] for r in table]);x=np.column_stack([np.ones(len(table)),[r[1] for r in table],[r[2] for r in table],[r[3] for r in table]]);coef=np.linalg.lstsq(x,y,rcond=None)[0];regression.append({"model":model,"task":task,"n_blocks":len(table),"intercept":coef[0],"TRAIN_utility_coefficient":coef[1],"persistence_margin_coefficient":coef[2],"block_rank_coefficient":coef[3],"descriptive_only":True})
    write_csv("PERSISTENCE_CONDITIONAL_REGRESSION.csv",regression)
    status = "CPU_EXPLORATORY_NON_EQUIVALENT" if CPU_EXPLORATORY else "COMPLETE_REPLAY_UNVERIFIED"
    report = ["# PU vs U-only explanatory analysis", "", f"Status: `{status}`. No neural network was trained; selectors and frozen checkpoint provenance were unchanged.", "", "## Required interpretation", "", "The CSV outputs contain the per-cell, subject-unit, and biological-subject bootstrap summaries required to distinguish overlap/persistence rediscovery from distinct subspaces. `UTILITY_MATCHED_TASK_SUMMARY.csv` is the primary conditional persistence test; block rows are aggregated within biological subject before bootstrap.", "", "Do not interpret P-only as a success criterion; it is retained only as the Experiment 1 negative control."]
    (OUT / "FINAL_PU_U_INTERPRETATION_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("PU_U_INTERPRETATION_AGGREGATE_COMPLETE",flush=True)


def _groups(rows: list[dict], key):
    out=defaultdict(list)
    for row in rows: out[key(row)].append(row)
    return out


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    one=sub.add_parser("cell");one.add_argument("model",choices=MODELS);one.add_argument("task",choices=TASKS);one.add_argument("fold",type=int,choices=range(5));one.add_argument("seed",type=int,choices=range(3)); all_cells=sub.add_parser("aggregate");all_cells.add_argument("--models",nargs="+",choices=MODELS,default=list(MODELS))
    args=parser.parse_args()
    if args.command=="cell": cell(args.model,args.task,args.fold,args.seed)
    else: aggregate(tuple(args.models))


if __name__ == "__main__": main()
