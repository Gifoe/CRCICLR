from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluation import SubjectEvaluator, subject_action_rows


def config_id(action: str, config: dict[str, object]) -> str:
    return action + "-" + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def action_grid(config: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    t3a = config["t3a"]
    t3a_grid = [dict(filter_k=int(k), confidence_threshold=c, prototype_interpolation=float(rho))
                for k, c, rho in itertools.product(t3a["filter_k"], t3a["confidence_threshold"], t3a["prototype_interpolation"])]
    adapter = config["adapter"]; fixed = config["fixed"]
    adapter_grid = [dict(steps=int(steps), learning_rate=float(lr), source_preservation_weight=float(beta),
                         reliability_quantile=float(q), consistency_weight=float(fixed["consistency_weight"]),
                         parameter_weight=float(fixed["parameter_weight"]), collapse_weight=float(fixed["collapse_weight"]),
                         collapse_rho=float(fixed["collapse_rho"]))
                    for steps, lr, beta, q in itertools.product(adapter["steps"], adapter["learning_rate"],
                                                               adapter["source_preservation_weight"], adapter["reliability_quantile"])]
    return {"official_t3a": t3a_grid, "robust_residual_adapter": adapter_grid}


def summarize(rows: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    keys = ["dataset", "seed", "action", "config_id", "stage"]
    records = []
    for values, group in rows.groupby(keys, sort=True):
        valid = group.action_available & (group.classification_degradation <= epsilon)
        subject_alpha = group.assign(safe_gain=np.where(valid, group.oracle_gain, 0.0))
        records.append(dict(zip(keys, values)) | {
            "n_subjects": int(group.subject_id.nunique()), "n_rows": int(len(group)),
            "availability_rate": float(group.action_available.mean()),
            "noninferiority_rate": float((group.classification_degradation <= epsilon).mean()),
            "mean_safe_gain": float(subject_alpha.safe_gain.mean()),
            "median_safe_gain": float(subject_alpha.safe_gain.median()),
            "positive_subject_rate": float(subject_alpha.groupby("subject_id").safe_gain.mean().gt(0).mean()),
            "mean_relative_gain": float(np.where(valid, group.relative_gain, 0.0).mean()),
            "harm_rate": float((group.classification_degradation > epsilon).mean()),
        })
    return pd.DataFrame(records)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); part = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(part, index=False); os.replace(part, path)


def run_action_search(root: str | Path, config: dict[str, object], device: str = "cuda", resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root); out = root / "outputs/v3_probecert/action_search"; out.mkdir(parents=True, exist_ok=True)
    detail_path = out / "ACTION_CONFIG_SUBJECT_RESULTS.parquet"
    detail = pd.read_parquet(detail_path) if resume and detail_path.exists() else pd.DataFrame()
    completed = set(zip(detail.get("dataset", []), detail.get("seed", []), detail.get("subject_id", []),
                        detail.get("action", []), detail.get("config_id", []), detail.get("stage", [])))
    rows = detail.to_dict("records"); grids = action_grid(config); first_n = int(config["first_stage_subjects"])
    survivors = int(config["survivors_per_action"]); epsilon = float(config["epsilon"])
    for dataset in ("hmc", "eegmmidb"):
        for seed in range(5):
            split = json.loads((root / "data/splits" / dataset / f"seed_{seed}.json").read_text())
            subjects = sorted(split["roles"]["meta_risk_train"])
            episodes = pd.read_parquet(root / "data/episodes_v3" / dataset / f"seed_{seed}.parquet").set_index("subject_id")
            evaluator = SubjectEvaluator(root, dataset, seed, device)
            cached = {subject: evaluator.prepare_episode(episodes.loc[subject]) for subject in subjects}
            for action, candidates in grids.items():
                for candidate in candidates:
                    cid = config_id(action, candidate)
                    for subject in subjects[:first_n]:
                        key = (dataset, seed, subject, action, cid, "screen")
                        if key in completed: continue
                        episode = cached[subject]; source = evaluator.source(episode)
                        result = evaluator.action(episode, action, candidate)
                        current = subject_action_rows(dataset, seed, subject, action, source["future"], result["future"],
                                                      episode["labels"], available=result["available"], status=result["status"], config_id=cid)
                        for row in current: row.update({"stage": "screen", "config_json": json.dumps(candidate, sort_keys=True)})
                        rows.extend(current); completed.add(key); _atomic_parquet(pd.DataFrame(rows), detail_path)
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                screening = summarize(pd.DataFrame(rows), epsilon)
                subset = screening[(screening.dataset == dataset) & (screening.seed == seed) &
                                   (screening.action == action) & (screening.stage == "screen")]
                selected_ids = subset.sort_values(["mean_safe_gain", "availability_rate", "harm_rate", "config_id"],
                                                  ascending=[False, False, True, True]).head(survivors).config_id.tolist()
                lookup = {config_id(action, candidate): candidate for candidate in candidates}
                for cid in selected_ids:
                    candidate = lookup[cid]
                    for subject in subjects:
                        key = (dataset, seed, subject, action, cid, "full")
                        if key in completed: continue
                        episode = cached[subject]; source = evaluator.source(episode)
                        result = evaluator.action(episode, action, candidate)
                        current = subject_action_rows(dataset, seed, subject, action, source["future"], result["future"],
                                                      episode["labels"], available=result["available"], status=result["status"], config_id=cid)
                        for row in current: row.update({"stage": "full", "config_json": json.dumps(candidate, sort_keys=True)})
                        rows.extend(current); completed.add(key); _atomic_parquet(pd.DataFrame(rows), detail_path)
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
    detail = pd.DataFrame(rows); summary = summarize(detail, epsilon)
    summary.to_csv(out / "ACTION_CONFIG_RESULTS.csv", index=False)
    chosen = []
    for (dataset, seed, action), group in summary[summary.stage == "full"].groupby(["dataset", "seed", "action"]):
        winner = group.sort_values(["mean_safe_gain", "availability_rate", "harm_rate", "config_id"],
                                   ascending=[False, False, True, True]).iloc[0]
        source = detail[(detail.config_id == winner.config_id) & (detail.stage == "full")].iloc[0]
        chosen.append({"dataset": dataset, "seed": int(seed), "action": action, "config_id": winner.config_id,
                       "config": json.loads(source.config_json), "selection_scope": "meta_risk_train_only",
                       "mean_safe_gain": float(winner.mean_safe_gain), "availability_rate": float(winner.availability_rate),
                       "harm_rate": float(winner.harm_rate)})
    (out / "SELECTED_ACTION_CONFIGS.json").write_text(json.dumps(chosen, indent=2, sort_keys=True) + "\n")
    return detail, summary
