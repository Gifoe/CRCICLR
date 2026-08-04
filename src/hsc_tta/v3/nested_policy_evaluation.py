from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hsc_tta.prediction_sets import evaluate_prediction_sets

from .access import EpisodeAccessController
from .action_search import action_grid, config_id
from .actions import source_probabilities
from .augmentations import TokenNuisanceAugmenter, nuisance_config_for
from .evaluation import LAMBDAS, SubjectEvaluator, safe_point, subject_action_rows
from .policy_certificate import calibrate_policy_index, joint_critical_index
from .policy_search import select_thresholds
from .probe_metrics import compute_probe_diagnostics, normalized_set_efficiency
from .probe_policy import ProbePolicy, ProbeThresholds


def _atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); part = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(part, index=False); os.replace(part, path)


def _hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _score(rows: list[dict[str, object]], epsilon: float) -> float:
    frame = pd.DataFrame(rows); valid = frame.action_available & (frame.classification_degradation <= epsilon)
    return float(np.where(valid, frame.oracle_gain, 0.0).mean())


def nested_action_selection(evaluator: SubjectEvaluator, episodes: pd.DataFrame, subjects: list[str],
                            search_config: dict[str, object], epsilon: float,
                            cache: dict[tuple[str, str], list[dict[str, object]]]) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    grids = action_grid(search_config); first = subjects[:int(search_config["first_stage_subjects"])]
    survivors = int(search_config["survivors_per_action"]); log = []; prepared = {}

    def evaluate(subject: str, action: str, candidate: dict[str, object]) -> list[dict[str, object]]:
        cid = config_id(action, candidate); key = (subject, cid)
        if key not in cache:
            episode = prepared.setdefault(subject, evaluator.prepare_episode(episodes.loc[subject]))
            source = evaluator.source(episode); result = evaluator.action(episode, action, candidate)
            cache[key] = subject_action_rows(evaluator.dataset, evaluator.seed, subject, action, source["future"],
                                              result["future"], episode["labels"], available=result["available"],
                                              status=result["status"], config_id=cid)
        return cache[key]

    selected = {}
    for action, candidates in grids.items():
        screening = []
        for candidate in candidates:
            rows = [row for subject in first for row in evaluate(subject, action, candidate)]
            score = _score(rows, epsilon); cid = config_id(action, candidate)
            screening.append((score, cid, candidate)); log.append({"action": action, "config_id": cid, "stage": "screen",
                "mean_safe_gain": score, "n_subjects": len(first), "config_json": json.dumps(candidate, sort_keys=True)})
        advanced = sorted(screening, key=lambda x: (-x[0], x[1]))[:survivors]; finals = []
        for _, cid, candidate in advanced:
            rows = [row for subject in subjects for row in evaluate(subject, action, candidate)]
            score = _score(rows, epsilon); finals.append((score, cid, candidate)); log.append({"action": action,
                "config_id": cid, "stage": "full", "mean_safe_gain": score, "n_subjects": len(subjects),
                "config_json": json.dumps(candidate, sort_keys=True)})
        score, cid, candidate = sorted(finals, key=lambda x: (-x[0], x[1]))[0]
        selected[action] = {"config_id": cid, "config": candidate, "meta_mean_safe_gain": score}
    return selected, log


def _diagnose(evaluator: SubjectEvaluator, episode: dict[str, object], action: str, selection: dict[str, object],
              augmenter: TokenNuisanceAugmenter) -> tuple[dict[str, object], dict[str, object]]:
    source = evaluator.source(episode); augmented = augmenter.all(episode["probe"])
    source_augmented = {name: source_probabilities(evaluator.model, tokens, evaluator.device) for name, tokens in augmented.items()}
    result = evaluator.action(episode, action, selection["config"], augmented)
    magnitude = float(result["diagnostics"].get("normalized_update_magnitude", np.inf))
    probe = compute_probe_diagnostics(source["probe"], result["probe"], list(source_augmented.values()),
                                      list(result["probe_augmented"].values()), LAMBDAS,
                                      action_available=result["available"], normalized_update_magnitude=magnitude)
    row = {"subject_id": episode["subject_id"], "action": action, "action_cost": 1 if action == "official_t3a" else 2,
           "dataset": evaluator.dataset, "seed": evaluator.seed,
           "config_id": selection["config_id"], "action_state_hash": result["state_hash"],
           "source_probe_efficiency": normalized_set_efficiency(source["probe"], LAMBDAS),
           "action_probe_efficiency": normalized_set_efficiency(result["probe"], LAMBDAS), **asdict(probe)}
    return row, result


def _open_future(evaluator: SubjectEvaluator, episode_row, episode: dict[str, object], results: dict[str, dict[str, object]],
                 controller: EpisodeAccessController) -> tuple[dict[str, np.ndarray], np.ndarray]:
    controller.access_future("future_inputs_handle", "future_labels_handle")
    evaluator.open_future(episode_row, episode); source = evaluator.source(episode); probabilities = {"no_tta": source["future"]}
    for action, result in results.items(): probabilities[action] = evaluator.predict_open_future(result, episode["future"])
    return probabilities, episode["labels"]


def _curves(probabilities: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, list[dict[str, float]]]:
    return {action: evaluate_prediction_sets(probability, labels, LAMBDAS) for action, probability in probabilities.items()}


def run_nested_policy_evaluation(root: str | Path, main: dict[str, object], search_config: dict[str, object],
                                 policy_grid: dict[str, object], device: str = "cuda", resume: bool = True,
                                 datasets: list[str] | None = None, seeds: list[int] | None = None,
                                 output_tag: str | None = None) -> dict[str, pd.DataFrame]:
    root = Path(root); out = root / "outputs/v3_probecert/nested_dev"
    if output_tag: out = out / "parts" / output_tag
    def existing(path: Path) -> list[dict[str, object]]:
        return pd.read_parquet(path).to_dict("records") if resume and path.exists() else []
    decisions=existing(out/"decisions/POLICY_DECISIONS.parquet")
    calibration=existing(out/"calibration/CALIBRATION_JOINT_INDICES.parquet")
    counterfactuals=existing(out/"counterfactuals/OUTER_COUNTERFACTUALS.parquet")
    fold_path=out/"metrics/RESULTS_BY_FOLD.csv"; fold_rows=(pd.read_csv(fold_path).to_dict("records") if resume and fold_path.exists() else [])
    search_path=out/"NESTED_ACTION_CONFIG_RESULTS.parquet"; search_rows=existing(search_path)
    completed_folds={(str(row["dataset"]),int(row["seed"]),int(row["outer_fold"])) for row in fold_rows}
    epsilon=float(main["epsilon"]); delta=float(main["delta"])
    for dataset in (datasets or main["datasets"]):
        augmenter = TokenNuisanceAugmenter(nuisance_config_for(dataset))
        for seed in (seeds if seeds is not None else main["seeds"]):
            evaluator = SubjectEvaluator(root, dataset, int(seed), device)
            episodes = pd.read_parquet(root / "data/episodes_v3" / dataset / f"seed_{seed}.parquet").set_index("subject_id")
            search_cache: dict[tuple[str, str], list[dict[str, object]]] = {}
            for split_path in sorted((root / "data/splits_v3_dev" / dataset / f"seed_{seed}").glob("outer_fold_*.json")):
                split=json.loads(split_path.read_text()); fold=int(split["outer_fold"]); meta=sorted(split["meta_fit_subjects"])
                if (dataset,int(seed),fold) in completed_folds: continue
                cal=sorted(split["calibration_subjects"]); outer=sorted(split["outer_evaluation_subjects"])
                selected, log = nested_action_selection(evaluator, episodes, meta, search_config, epsilon, search_cache)
                for row in log: row.update({"dataset":dataset,"seed":int(seed),"outer_fold":fold})
                search_rows.extend(log)
                meta_diag=[]; meta_out=[]
                for subject in meta:
                    episode=evaluator.prepare_episode(episodes.loc[subject]) ; source=evaluator.source(episode)
                    for action, selection in selected.items():
                        diag,result=_diagnose(evaluator,episode,action,selection,augmenter); meta_diag.append(diag)
                        meta_out.extend(subject_action_rows(dataset,int(seed),subject,action,source["future"],result["future"],
                                                            episode["labels"],available=result["available"],status=result["status"],
                                                            config_id=selection["config_id"]))
                meta_diag=pd.DataFrame(meta_diag); meta_out=pd.DataFrame(meta_out)
                for alpha in main["alphas"]:
                    thresholds, threshold_search=select_thresholds(meta_diag,meta_out[meta_out.alpha==alpha],policy_grid,epsilon)
                    policy=ProbePolicy(thresholds); method_hash=_hash({"dataset":dataset,"seed":seed,"fold":fold,"alpha":alpha,
                        "actions":selected,"nuisance":asdict(augmenter.config),"thresholds":asdict(thresholds),
                        "lambda_grid":LAMBDAS.tolist(),"epsilon":epsilon,"delta":delta})
                    cal_indices=[]; no_indices=[]
                    for subject in cal:
                        controller=EpisodeAccessController(subject,method_hash); episode=evaluator.prepare_episode(episodes.loc[subject],include_future=False)
                        diag=[]; action_results={}
                        for action,selection in selected.items():
                            row,result=_diagnose(evaluator,episode,action,selection,augmenter); diag.append(row); action_results[action]=result
                        controller.begin_probe({"no_tta":"source-model",**{a:r["state_hash"] for a,r in action_results.items()}})
                        choice=policy.decide(pd.DataFrame(diag)); action=choice["selected_action"]
                        decision={"subject_id":subject,"selected_action":action,"action_state_hash":("source-model" if action=="no_tta" else action_results[action]["state_hash"]),
                                  "config_hash":method_hash,"lambda_index":-1,"role":"calibration","alpha":alpha}
                        path=out/"decisions"/dataset/f"seed_{seed}"/f"fold_{fold}"/f"alpha_{alpha}"/f"{subject.replace(':','_')}.json"
                        controller.freeze_decision(decision,path); probabilities,labels=_open_future(evaluator,episodes.loc[subject],episode,action_results,controller)
                        decisions.append({**decision,**choice,"dataset":dataset,"seed":int(seed),"outer_fold":fold,
                                          "decision_hash":controller.decision_hash,"decision_path":str(path)})
                        curves=_curves(probabilities,labels); selected_curve=curves[action]; degradation=float(selected_curve[0]["argmax_error"]-curves["no_tta"][0]["argmax_error"])
                        risks=np.asarray([x["future_risk"] for x in selected_curve]); j=joint_critical_index(risks,degradation,alpha=float(alpha),epsilon=epsilon,sentinel_index=20)
                        no_j=joint_critical_index(np.asarray([x["future_risk"] for x in curves["no_tta"]]),0,alpha=float(alpha),epsilon=epsilon,sentinel_index=20)
                        cal_indices.append(j); no_indices.append(no_j); calibration.append({"dataset":dataset,"seed":int(seed),"outer_fold":fold,"alpha":alpha,
                            "subject_id":subject,"policy":"probecert_v3","joint_index":j,"selected_action":action,"degradation":degradation,"decision_hash":controller.decision_hash})
                        calibration.append({"dataset":dataset,"seed":int(seed),"outer_fold":fold,"alpha":alpha,
                            "subject_id":subject,"policy":"no_tta_global_crc","joint_index":no_j,"selected_action":"no_tta","degradation":0.0,"decision_hash":controller.decision_hash})
                    calibrated=calibrate_policy_index(np.asarray(cal_indices),delta=delta,sentinel_index=20)
                    no_calibrated=calibrate_policy_index(np.asarray(no_indices),delta=delta,sentinel_index=20)
                    for subject in outer:
                        controller=EpisodeAccessController(subject,method_hash); episode=evaluator.prepare_episode(episodes.loc[subject],include_future=False)
                        diag=[]; action_results={}
                        for action,selection in selected.items():
                            row,result=_diagnose(evaluator,episode,action,selection,augmenter); diag.append(row); action_results[action]=result
                        controller.begin_probe({"no_tta":"source-model",**{a:r["state_hash"] for a,r in action_results.items()}})
                        choice=policy.decide(pd.DataFrame(diag)); action=choice["selected_action"]
                        decision={"subject_id":subject,"selected_action":action,"action_state_hash":("source-model" if action=="no_tta" else action_results[action]["state_hash"]),
                                  "config_hash":method_hash,"lambda_index":calibrated.lambda_index,"role":"outer_evaluation","alpha":alpha}
                        path=out/"decisions"/dataset/f"seed_{seed}"/f"fold_{fold}"/f"alpha_{alpha}"/f"{subject.replace(':','_')}.json"
                        controller.freeze_decision(decision,path); probabilities,labels=_open_future(evaluator,episodes.loc[subject],episode,action_results,controller)
                        decisions.append({**decision,**choice,"dataset":dataset,"seed":int(seed),"outer_fold":fold,
                                          "decision_hash":controller.decision_hash,"decision_path":str(path)})
                        curves=_curves(probabilities,labels); source_error=curves["no_tta"][0]["argmax_error"]
                        for policy_name,current_action,index in [("probecert_v3",action,calibrated.lambda_index),("no_tta_global_crc","no_tta",no_calibrated.lambda_index)]:
                            point=curves[current_action][index]; degradation=float(point["argmax_error"]-source_error)
                            counterfactuals.append({"dataset":dataset,"seed":int(seed),"outer_fold":fold,"alpha":alpha,"subject_id":subject,"policy":policy_name,
                                "selected_action":current_action,"lambda_index":index,"future_risk":point["future_risk"],"average_set_size":point["average_set_size"],
                                "singleton_rate":point["singleton_rate"],"argmax_error":point["argmax_error"],"macro_f1":point["macro_f1"],
                                "degradation":degradation,"joint_violation":bool(point["future_risk"]>alpha or degradation>epsilon),"sentinel":index==20,
                                "intervention":current_action!="no_tta","method_hash":method_hash})
                    current=pd.DataFrame(counterfactuals); subset=current[(current.dataset==dataset)&(current.seed==seed)&(current.outer_fold==fold)&(current.alpha==alpha)]
                    for policy_name,group in subset.groupby("policy"):
                        fold_rows.append({"dataset":dataset,"seed":int(seed),"outer_fold":fold,"alpha":alpha,"policy":policy_name,"n_subjects":group.subject_id.nunique(),
                            "joint_violation_rate":group.joint_violation.mean(),"joint_validity":1-group.joint_violation.mean(),"average_set_size":group.average_set_size.mean(),
                            "singleton_rate":group.singleton_rate.mean(),"argmax_error":group.argmax_error.mean(),"macro_f1":group.macro_f1.mean(),
                            "intervention_rate":group.intervention.mean(),"sentinel_rate":group.sentinel.mean(),
                            "calibration_insufficient":calibrated.insufficient if policy_name=="probecert_v3" else no_calibrated.insufficient})
                _atomic(pd.DataFrame(decisions),out/"decisions/POLICY_DECISIONS.parquet") if decisions else None
                _atomic(pd.DataFrame(calibration),out/"calibration/CALIBRATION_JOINT_INDICES.parquet")
                _atomic(pd.DataFrame(counterfactuals),out/"counterfactuals/OUTER_COUNTERFACTUALS.parquet")
                fold_path.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(fold_rows).to_csv(fold_path,index=False)
                _atomic(pd.DataFrame(search_rows),search_path)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"calibration":pd.DataFrame(calibration),"counterfactuals":pd.DataFrame(counterfactuals),
            "folds":pd.DataFrame(fold_rows),"action_search":pd.DataFrame(search_rows)}
