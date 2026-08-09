from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from hsc_tta.prediction_sets import evaluate_prediction_sets

from .access import EpisodeAccessController
from .augmentations import TokenNuisanceAugmenter, nuisance_config_for
from .evaluation import LAMBDAS, SubjectEvaluator
from .nested_policy_evaluation import _diagnose, _hash, _open_future


def build_nested_surfaces(root: str | Path, main: dict[str, object], dataset: str, seed: int,
                          output_tag: str, device: str = "cuda") -> pd.DataFrame:
    root=Path(root); nested=root/"outputs/v3_probecert/nested_dev"; search=pd.read_parquet(nested/"NESTED_ACTION_CONFIG_RESULTS.parquet")
    current=search[(search.dataset==dataset)&(search.seed==seed)&(search.stage=="full")]
    episodes=pd.read_parquet(root/"data/episodes_v3"/dataset/f"seed_{seed}.parquet").set_index("subject_id")
    evaluator=SubjectEvaluator(root,dataset,seed,device); augmenter=TokenNuisanceAugmenter(nuisance_config_for(dataset))
    out=nested/"surfaces";out.mkdir(parents=True,exist_ok=True);output=out/f"NESTED_ACTION_SURFACES_{output_tag}.parquet"
    existing=pd.read_parquet(output) if output.exists() else pd.DataFrame();rows=existing.to_dict("records")
    completed=set(existing.outer_fold.unique()) if len(existing) else set()
    for split_path in sorted((root/"data/splits_v3_dev"/dataset/f"seed_{seed}").glob("outer_fold_*.json")):
        split=json.loads(split_path.read_text()); fold=int(split["outer_fold"]); fold_search=current[current.outer_fold==fold]
        if fold in completed: continue
        selected={}
        for action,group in fold_search.groupby("action"):
            winner=group.sort_values(["mean_safe_gain","config_id"],ascending=[False,True]).iloc[0]
            selected[action]={"config_id":winner.config_id,"config":json.loads(winner.config_json)}
        roles={subject:role for role in ("meta_fit_subjects","calibration_subjects","outer_evaluation_subjects") for subject in split[role]}
        for subject,role in sorted(roles.items()):
            include_future=role=="meta_fit_subjects"; episode=evaluator.prepare_episode(episodes.loc[subject],include_future=include_future)
            diagnostics=[]; results={}
            for action,selection in selected.items():
                diag,result=_diagnose(evaluator,episode,action,selection,augmenter); diagnostics.append(diag); results[action]=result
            if not include_future:
                method_hash=_hash({"surface":"frozen_counterfactual","dataset":dataset,"seed":seed,"fold":fold,"actions":selected})
                controller=EpisodeAccessController(subject,method_hash); controller.begin_probe({"no_tta":"source-model",**{a:r["state_hash"] for a,r in results.items()}})
                decision={"subject_id":subject,"selected_action":"no_tta","action_state_hash":"source-model","config_hash":method_hash,"lambda_index":20,"role":role}
                path=nested/"surface_freezes"/dataset/f"seed_{seed}"/f"fold_{fold}"/f"{subject.replace(':','_')}.json"
                controller.freeze_decision(decision,path); probabilities,labels=_open_future(evaluator,episodes.loc[subject],episode,results,controller)
            else:
                source=evaluator.source(episode); probabilities={"no_tta":source["future"],**{a:r["future"] for a,r in results.items()}}; labels=episode["labels"]
            source_curve=evaluate_prediction_sets(probabilities["no_tta"],labels,LAMBDAS)
            source_probe=evaluator.source(episode)["probe"]
            for diag in diagnostics:
                action=diag["action"]; curve=evaluate_prediction_sets(probabilities[action],labels,LAMBDAS)
                action_probe=results[action]["probe"]
                entropy=lambda p: float(np.mean(-(p*np.log(np.maximum(p,1e-12))).sum(1)))
                row={**diag,"dataset":dataset,"seed":seed,"outer_fold":fold,"role":role,
                     "entropy_change":entropy(action_probe)-entropy(source_probe),
                     "prediction_agreement":float(np.mean(action_probe.argmax(1)==source_probe.argmax(1))),
                     "argmax_error":curve[0]["argmax_error"],"source_argmax_error":source_curve[0]["argmax_error"],
                     "degradation":curve[0]["argmax_error"]-source_curve[0]["argmax_error"],"macro_f1":curve[0]["macro_f1"]}
                for j,(point,source_point) in enumerate(zip(curve,source_curve)):
                    row[f"risk_j{j}"]=point["future_risk"]; row[f"size_j{j}"]=point["average_set_size"]; row[f"singleton_j{j}"]=point["singleton_rate"]
                    row[f"source_risk_j{j}"]=source_point["future_risk"]; row[f"source_size_j{j}"]=source_point["average_set_size"]
                rows.append(row)
        pd.DataFrame(rows).to_parquet(output,index=False)
    return pd.DataFrame(rows)
