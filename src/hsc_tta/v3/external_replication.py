from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .access import EpisodeAccessController
from .augmentations import TokenNuisanceAugmenter, nuisance_config_for
from .evaluation import SubjectEvaluator
from .nested_policy_evaluation import _curves, _diagnose, _hash, _open_future
from .policy_certificate import calibrate_policy_index, joint_critical_index
from .policy_search import select_thresholds
from .probe_policy import ProbePolicy


def run_cap_replication(root: str | Path, main: dict[str, object], policy_grid: dict[str, object],
                        device: str = "cuda", seeds: list[int] | None = None,
                        output_tag: str | None = None) -> pd.DataFrame:
    root=Path(root); base=root/"outputs/v3_probecert/external_site"
    out=base if output_tag is None else base/"parts"/output_tag
    out.mkdir(parents=True,exist_ok=True)
    development=root/"outputs/v3_probecert/cross_context_surfaces"
    diagnostics=pd.read_parquet(development/"PROBE_DIAGNOSTICS.parquet")
    outcomes=pd.read_parquet(development/"META_FUTURE_ACTION_OUTCOMES.parquet")
    selected=json.loads((root/"outputs/v3_probecert/action_search/SELECTED_ACTION_CONFIGS.json").read_text())
    configs={(int(x["seed"]),x["action"]):x for x in selected if x["dataset"]=="hmc"}
    augmenter=TokenNuisanceAugmenter(nuisance_config_for("cap")); rows=[]; calibration_rows=[]
    selected_seeds=[int(x) for x in (main["seeds"] if seeds is None else seeds)]
    for seed in selected_seeds:
        split=json.loads((root/"data/splits/cap"/f"seed_{seed}.json").read_text())["roles"]
        episodes=pd.read_parquet(root/"data/episodes_v3/cap"/f"seed_{seed}.parquet").set_index("subject_id")
        evaluator=SubjectEvaluator(root,"cap",int(seed),device,model_dataset="hmc")
        action_configs={a:configs[(int(seed),a)] for a in ("official_t3a","robust_residual_adapter")}
        hmc_diag=diagnostics[(diagnostics.dataset=="hmc")&(diagnostics.seed==seed)]
        for alpha in main["alphas"]:
            hmc_out=outcomes[(outcomes.dataset=="hmc")&(outcomes.seed==seed)&(outcomes.alpha==alpha)]
            thresholds,_=select_thresholds(hmc_diag,hmc_out,policy_grid,float(main["epsilon"])); policy=ProbePolicy(thresholds)
            method_hash=_hash({"transfer":"hmc_to_cap","seed":seed,"alpha":alpha,"actions":action_configs,
                               "thresholds":asdict(thresholds),"nuisance":asdict(augmenter.config),
                               "site_specific_component":"conformal_quantile_only"})
            indices=[]; no_indices=[]
            for subject in sorted(split["target_site_calibration"]):
                controller=EpisodeAccessController(subject,method_hash); episode=evaluator.prepare_episode(episodes.loc[subject],include_future=False)
                diag=[]; results={}
                for action,selection in action_configs.items():
                    current,result=_diagnose(evaluator,episode,action,selection,augmenter); diag.append(current); results[action]=result
                controller.begin_probe({"no_tta":"source-model",**{a:r["state_hash"] for a,r in results.items()}})
                choice=policy.decide(pd.DataFrame(diag)); action=choice["selected_action"]
                decision={"subject_id":subject,"selected_action":action,"action_state_hash":("source-model" if action=="no_tta" else results[action]["state_hash"]),
                          "config_hash":method_hash,"lambda_index":-1,"role":"cap_calibration","alpha":alpha}
                path=out/"decisions"/f"seed_{seed}"/f"alpha_{alpha}"/f"{subject.replace(':','_')}.json"; controller.freeze_decision(decision,path)
                probabilities,labels=_open_future(evaluator,episodes.loc[subject],episode,results,controller); curves=_curves(probabilities,labels)
                degradation=curves[action][0]["argmax_error"]-curves["no_tta"][0]["argmax_error"]
                index=joint_critical_index(np.asarray([x["future_risk"] for x in curves[action]]),degradation,alpha=float(alpha),epsilon=float(main["epsilon"]),sentinel_index=20)
                no_index=joint_critical_index(np.asarray([x["future_risk"] for x in curves["no_tta"]]),0,alpha=float(alpha),epsilon=float(main["epsilon"]),sentinel_index=20)
                indices.append(index); no_indices.append(no_index); calibration_rows.append({"seed":seed,"alpha":alpha,"subject_id":subject,
                    "joint_index":index,"no_tta_joint_index":no_index,"selected_action":action})
            calibrated=calibrate_policy_index(np.asarray(indices),delta=float(main["delta"]),sentinel_index=20)
            no_calibrated=calibrate_policy_index(np.asarray(no_indices),delta=float(main["delta"]),sentinel_index=20)
            for subject in sorted(split["external_final_test"]):
                controller=EpisodeAccessController(subject,method_hash); episode=evaluator.prepare_episode(episodes.loc[subject],include_future=False)
                diag=[]; results={}
                for action,selection in action_configs.items():
                    current,result=_diagnose(evaluator,episode,action,selection,augmenter); diag.append(current); results[action]=result
                controller.begin_probe({"no_tta":"source-model",**{a:r["state_hash"] for a,r in results.items()}})
                choice=policy.decide(pd.DataFrame(diag)); action=choice["selected_action"]
                decision={"subject_id":subject,"selected_action":action,"action_state_hash":("source-model" if action=="no_tta" else results[action]["state_hash"]),
                          "config_hash":method_hash,"lambda_index":calibrated.lambda_index,"role":"cap_external_replication","alpha":alpha}
                path=out/"decisions"/f"seed_{seed}"/f"alpha_{alpha}"/f"{subject.replace(':','_')}.json"; controller.freeze_decision(decision,path)
                probabilities,labels=_open_future(evaluator,episodes.loc[subject],episode,results,controller); curves=_curves(probabilities,labels); source_error=curves["no_tta"][0]["argmax_error"]
                for policy_name,current,index in [("probecert_v3",action,calibrated.lambda_index),("no_tta_global_crc","no_tta",no_calibrated.lambda_index)]:
                    point=curves[current][index]; degradation=point["argmax_error"]-source_error
                    rows.append({"dataset":"cap","seed":seed,"alpha":alpha,"subject_id":subject,"policy":policy_name,"selected_action":current,
                        "lambda_index":index,"future_risk":point["future_risk"],"average_set_size":point["average_set_size"],"singleton_rate":point["singleton_rate"],
                        "argmax_error":point["argmax_error"],"macro_f1":point["macro_f1"],"degradation":degradation,
                        "joint_violation":bool(point["future_risk"]>alpha or degradation>float(main["epsilon"])),"sentinel":index==20,"intervention":current!="no_tta"})
    frame=pd.DataFrame(rows); frame.to_parquet(out/"CAP_EXTERNAL_SUBJECT_RESULTS.parquet",index=False)
    pd.DataFrame(calibration_rows).to_parquet(out/"CAP_CALIBRATION_JOINT_INDICES.parquet",index=False)
    summary=frame.groupby(["dataset","seed","alpha","policy"],as_index=False).agg(joint_violation_rate=("joint_violation","mean"),
        average_set_size=("average_set_size","mean"),singleton_rate=("singleton_rate","mean"),argmax_error=("argmax_error","mean"),
        macro_f1=("macro_f1","mean"),intervention_rate=("intervention","mean"),sentinel_rate=("sentinel","mean"))
    summary.to_csv(out/"CAP_EXTERNAL_SUMMARY.csv",index=False); return frame


def merge_cap_parts(root: str | Path, expected_seeds: list[int]) -> pd.DataFrame:
    root=Path(root); out=root/"outputs/v3_probecert/external_site"; parts=out/"parts"
    result_files=sorted(parts.glob("*/CAP_EXTERNAL_SUBJECT_RESULTS.parquet"))
    calibration_files=sorted(parts.glob("*/CAP_CALIBRATION_JOINT_INDICES.parquet"))
    if not result_files or not calibration_files:
        raise FileNotFoundError("CAP part files are incomplete")
    frame=pd.concat([pd.read_parquet(path) for path in result_files],ignore_index=True)
    calibration=pd.concat([pd.read_parquet(path) for path in calibration_files],ignore_index=True)
    frame=frame.drop_duplicates(["seed","alpha","subject_id","policy"],keep="last")
    calibration=calibration.drop_duplicates(["seed","alpha","subject_id"],keep="last")
    observed=sorted(int(x) for x in frame.seed.unique())
    if observed != sorted(int(x) for x in expected_seeds):
        raise RuntimeError(f"CAP seed coverage mismatch: expected {expected_seeds}, observed {observed}")
    frame.to_parquet(out/"CAP_EXTERNAL_SUBJECT_RESULTS.parquet",index=False)
    calibration.to_parquet(out/"CAP_CALIBRATION_JOINT_INDICES.parquet",index=False)
    summary=frame.groupby(["dataset","seed","alpha","policy"],as_index=False).agg(joint_violation_rate=("joint_violation","mean"),
        average_set_size=("average_set_size","mean"),singleton_rate=("singleton_rate","mean"),argmax_error=("argmax_error","mean"),
        macro_f1=("macro_f1","mean"),intervention_rate=("intervention","mean"),sentinel_rate=("sentinel","mean"))
    summary.to_csv(out/"CAP_EXTERNAL_SUMMARY.csv",index=False)
    return frame
