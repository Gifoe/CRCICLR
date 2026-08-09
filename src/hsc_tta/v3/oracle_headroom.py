from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluation import SubjectEvaluator, subject_action_rows
from .statistics import average_seeds_by_subject, subject_cluster_bootstrap


def run_oracle_headroom(root: str | Path, config: dict[str, object], device: str = "cuda",
                        resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    root = Path(root); out = root / "outputs/v3_probecert/oracle_headroom"; out.mkdir(parents=True, exist_ok=True)
    selected = json.loads((root / "outputs/v3_probecert/action_search/SELECTED_ACTION_CONFIGS.json").read_text())
    configs = {(x["dataset"], int(x["seed"]), x["action"]): x for x in selected}
    path = out / "ORACLE_HEADROOM_BY_SUBJECT.parquet"
    frame = pd.read_parquet(path) if resume and path.exists() else pd.DataFrame(); rows = frame.to_dict("records")
    completed = set(zip(frame.get("dataset", []), frame.get("seed", []), frame.get("subject_id", []), frame.get("action", [])))
    epsilon_values = tuple(float(x) for x in config["epsilon_sensitivity"])
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            split = json.loads((root / "data/splits" / dataset / f"seed_{seed}.json").read_text())
            subjects = sorted(split["roles"]["meta_risk_train"])
            episodes = pd.read_parquet(root / "data/episodes_v3" / dataset / f"seed_{seed}.parquet").set_index("subject_id")
            evaluator = SubjectEvaluator(root, dataset, int(seed), device)
            cached = {subject: evaluator.prepare_episode(episodes.loc[subject]) for subject in subjects}
            for subject in subjects:
                episode = cached[subject]; source = evaluator.source(episode)
                for action in ("official_t3a", "robust_residual_adapter"):
                    key = (dataset, int(seed), subject, action)
                    if key in completed: continue
                    chosen = configs[(dataset, int(seed), action)]; result = evaluator.action(episode, action, chosen["config"])
                    current = subject_action_rows(dataset, int(seed), subject, action, source["future"], result["future"],
                                                  episode["labels"], available=result["available"], status=result["status"],
                                                  config_id=chosen["config_id"])
                    for row in current:
                        for eps in epsilon_values:
                            item = dict(row); item["epsilon"] = eps
                            item["eligible"] = bool(item["action_available"] and item["classification_degradation"] <= eps)
                            item["safe_oracle_gain"] = item["oracle_gain"] if item["eligible"] and item["oracle_gain"] > 0 else 0.0
                            rows.append(item)
                    completed.add(key); pd.DataFrame(rows).to_parquet(path, index=False)
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
    frame = pd.DataFrame(rows); records = []
    reps = int(config["bootstrap_repetitions"])
    for (dataset, alpha, epsilon), group in frame.groupby(["dataset", "alpha", "epsilon"]):
        best = group.sort_values("safe_oracle_gain").groupby(["seed", "subject_id"], as_index=False).tail(1)
        best["positive"] = (best.safe_oracle_gain > 0).astype(float)
        best["relative_reduction"] = best.safe_oracle_gain / best.source_safe_size.clip(lower=1e-12)
        averaged = average_seeds_by_subject(best, ["safe_oracle_gain", "relative_reduction", "positive"])
        ci = subject_cluster_bootstrap(averaged, "relative_reduction", repetitions=reps)
        action_contribution = best[best.safe_oracle_gain > 0].action.value_counts(normalize=True).to_dict()
        per_seed_positive = best.groupby("seed").positive.mean()
        single_action_rates = group.assign(positive_action=group.safe_oracle_gain > 0).groupby("action").positive_action.mean()
        records.append({"dataset": dataset, "alpha": alpha, "epsilon": epsilon,
                        "positive_subject_rate": float(averaged.positive.mean()),
                        "mean_gain": float(averaged.safe_oracle_gain.mean()),
                        "median_gain": float(averaged.safe_oracle_gain.median()),
                        "relative_set_size_reduction": float(averaged.relative_reduction.mean()),
                        "relative_ci_lower": ci["ci_lower"], "relative_ci_upper": ci["ci_upper"],
                        "non_tta_action_subject_rate": float(best.groupby(["seed", "subject_id"]).safe_oracle_gain.max().gt(0).mean()),
                        "maximum_single_action_positive_rate": float(single_action_rates.max()),
                        "harm_rate": float((group.classification_degradation > epsilon).mean()),
                        "minimum_seed_positive_rate": float(per_seed_positive.min()),
                        "official_t3a_contribution": float(action_contribution.get("official_t3a", 0)),
                        "robust_adapter_contribution": float(action_contribution.get("robust_residual_adapter", 0)),
                        "n_unique_subjects": int(averaged.subject_id.nunique())})
    summary = pd.DataFrame(records); summary.to_csv(out / "ORACLE_HEADROOM_SUMMARY.csv", index=False)
    seed_summary = frame.assign(positive=frame.safe_oracle_gain > 0,
                                harm=frame.classification_degradation > frame.epsilon).groupby(
        ["dataset", "seed", "alpha", "epsilon"], as_index=False).agg(
        positive_action_row_rate=("positive", "mean"), mean_safe_oracle_gain=("safe_oracle_gain", "mean"),
        harm_rate=("harm", "mean"))
    seed_summary.to_csv(out / "ORACLE_HEADROOM_BY_SEED.csv", index=False)
    main = summary[summary.epsilon == float(config["epsilon"])]
    per_dataset = main.groupby("dataset").agg(ci_lower=("relative_ci_lower", "min"), positive=("positive_subject_rate", "mean"),
                                               action_rate=("maximum_single_action_positive_rate", "max"),
                                               min_seed=("minimum_seed_positive_rate", "min"))
    go = bool(((per_dataset.ci_lower > 0) & (per_dataset.positive >= .20) &
               (per_dataset.action_rate >= .10) & (per_dataset.min_seed > 0)).any())
    (out / "ORACLE_GATE.json").write_text(json.dumps({"go": go, "criteria": {
        "some_primary_dataset_relative_ci_lower_gt_zero": bool((per_dataset.ci_lower > 0).any()),
        "positive_subject_rate_at_least_0.20": bool((per_dataset.positive >= .20).any()),
        "non_no_tta_action_benefits_at_least_0.10": bool((per_dataset.action_rate >= .10).any()),
        "not_single_seed_driven": bool((per_dataset.min_seed > 0).any())}}, indent=2, sort_keys=True) + "\n")
    return frame, summary, go
