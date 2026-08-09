from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .augmentations import TokenNuisanceAugmenter, nuisance_config_for
from .actions import source_probabilities
from .evaluation import LAMBDAS, SubjectEvaluator, subject_action_rows
from .probe_metrics import LOG2, compute_probe_diagnostics, jensen_shannon, normalized_set_efficiency


def _atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); part = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(part, index=False); os.replace(part, path)


def build_cross_context_surfaces(root: str | Path, config: dict[str, object], device: str = "cuda",
                                 resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root); out = root / "outputs/v3_probecert/cross_context_surfaces"; out.mkdir(parents=True, exist_ok=True)
    diagnostics_path = out / "PROBE_DIAGNOSTICS.parquet"; outcomes_path = out / "META_FUTURE_ACTION_OUTCOMES.parquet"
    diagnostics = pd.read_parquet(diagnostics_path) if resume and diagnostics_path.exists() else pd.DataFrame()
    outcomes = pd.read_parquet(outcomes_path) if resume and outcomes_path.exists() else pd.DataFrame()
    diagnostic_rows = diagnostics.to_dict("records"); outcome_rows = outcomes.to_dict("records")
    completed = set(zip(diagnostics.get("dataset", []), diagnostics.get("seed", []), diagnostics.get("subject_id", []), diagnostics.get("action", [])))
    chosen = json.loads((root / "outputs/v3_probecert/action_search/SELECTED_ACTION_CONFIGS.json").read_text())
    selected = {(x["dataset"], int(x["seed"]), x["action"]): x for x in chosen}
    augmentation_rows = []; augmentation_configs = {}
    for dataset in config["datasets"]:
        augmenter = TokenNuisanceAugmenter(nuisance_config_for(dataset))
        augmentation_configs[dataset] = asdict(augmenter.config)
        for seed in config["seeds"]:
            split = json.loads((root / "data/splits" / dataset / f"seed_{seed}.json").read_text())
            subjects = sorted(split["roles"]["meta_risk_train"])
            episodes = pd.read_parquet(root / "data/episodes_v3" / dataset / f"seed_{seed}.parquet").set_index("subject_id")
            evaluator = SubjectEvaluator(root, dataset, int(seed), device)
            for subject in subjects:
                episode = evaluator.prepare_episode(episodes.loc[subject]); source = evaluator.source(episode)
                augmented_tokens = augmenter.all(episode["probe"])
                source_augmented = {name: source_probabilities(evaluator.model, tokens, device)
                                    for name, tokens in augmented_tokens.items()}
                for name, probability in source_augmented.items():
                    augmentation_rows.append({"dataset": dataset, "seed": int(seed), "subject_id": subject,
                                              "augmentation": name,
                                              "normalized_js": float(jensen_shannon(source["probe"], probability).mean() / LOG2),
                                              "argmax_agreement": float(np.mean(source["probe"].argmax(1) == probability.argmax(1)))})
                for action in ("official_t3a", "robust_residual_adapter"):
                    key = (dataset, int(seed), subject, action)
                    if key in completed: continue
                    selection = selected[(dataset, int(seed), action)]
                    result = evaluator.action(episode, action, selection["config"], augmented_tokens)
                    magnitude = float(result["diagnostics"].get("normalized_update_magnitude", np.inf))
                    probe = compute_probe_diagnostics(source["probe"], result["probe"], list(source_augmented.values()),
                                                      list(result["probe_augmented"].values()), LAMBDAS,
                                                      action_available=result["available"], normalized_update_magnitude=magnitude)
                    row = {"dataset": dataset, "seed": int(seed), "subject_id": subject, "action": action,
                           "config_id": selection["config_id"], "config_json": json.dumps(selection["config"], sort_keys=True),
                           "action_status": result["status"], "action_available": result["available"],
                           "action_state_hash": result["state_hash"], "action_cost": 1 if action == "official_t3a" else 2,
                           "source_probe_efficiency": normalized_set_efficiency(source["probe"], LAMBDAS),
                           "action_probe_efficiency": normalized_set_efficiency(result["probe"], LAMBDAS),
                           **asdict(probe)}
                    diagnostic_rows.append(row)
                    current = subject_action_rows(dataset, int(seed), subject, action, source["future"], result["future"],
                                                  episode["labels"], available=result["available"], status=result["status"],
                                                  config_id=selection["config_id"])
                    outcome_rows.extend(current); completed.add(key)
                    _atomic(pd.DataFrame(diagnostic_rows), diagnostics_path); _atomic(pd.DataFrame(outcome_rows), outcomes_path)
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
    validation = pd.DataFrame(augmentation_rows)
    validation.groupby(["dataset", "augmentation"], as_index=False).agg(
        mean_normalized_js=("normalized_js", "mean"), p95_normalized_js=("normalized_js", lambda x: float(np.quantile(x, .95))),
        mean_argmax_agreement=("argmax_agreement", "mean")).to_csv(out / "AUGMENTATION_SOURCE_VALIDATION.csv", index=False)
    (out / "AUGMENTATION_CONFIG.json").write_text(json.dumps(augmentation_configs, indent=2, sort_keys=True) + "\n")
    return pd.DataFrame(diagnostic_rows), pd.DataFrame(outcome_rows)
