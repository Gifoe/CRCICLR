from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from hsc_tta.contextual_risk.io import atomic_json


def sha256_paths(paths: list[Path]) -> str:
    digest=hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def write_stage0_reports(repo:Path,summary:pd.DataFrame,passed:bool,decision:dict[str,Any])->None:
    delivery=repo/"delivery/budgeted_risk";stage=delivery/"stage0";stage.mkdir(parents=True,exist_ok=True)
    table=summary.to_markdown(index=False,floatfmt=".4f")
    (stage/"FULL_CONTEXT_UPPER_BOUND.md").write_text(
        "# Stage-0A full-context upper bound\n\n"
        "Metrics are subject-level after averaging the five source-head seeds. The gate was applied separately to each dataset.\n\n"
        +table+"\n\nDecision: **"+("FULL_CONTEXT_GO" if passed else "FULL_CONTEXT_NO_GO")+"**.\n",encoding="utf-8")
    not_run="Not run because the preregistered full-context gate failed. No budget or acquisition result was generated.\n"
    (stage/"RANDOM_BUDGET_BASELINE.md").write_text("# Stage-0B random/budget baseline\n\n"+("Pending Stage-0A pass.\n" if passed else not_run),encoding="utf-8")
    (stage/"ACQUISITION_COMPARISON.md").write_text("# Stage-0C acquisition comparison\n\n"+("Pending Stage-0B pass.\n" if passed else not_run),encoding="utf-8")
    verdict="STAGE0_GO" if passed else "STAGE0_NO_GO"
    (stage/"STAGE0_DECISION.md").write_text(
        f"# Stage-0 decision\n\nVerdict: **{verdict}**.\n\n{table}\n\n"
        +("The full-context feasibility prerequisite passed.\n" if passed else
          "The full-context feasibility prerequisite failed for at least one internal dataset. The hard-stop rule was applied before budget experiments, formal calibration, internal final evaluation, and CAP.\n"),encoding="utf-8")
    atomic_json(decision,stage/"STAGE0_DECISION.json")
    if not passed:
        (delivery/"STOP_BUDGETED_CALIBRATION.md").write_text(
            "# STOP: budgeted subject-risk calibration\n\n"
            "Stage-0A did not meet the preregistered feasibility gate. Budget experiments were not opened.\n\n"
            "- formal_calibration_opened: false\n- internal_final_opened: false\n- cap_opened: false\n",encoding="utf-8")


def plot_full_context(repo:Path,results:pd.DataFrame)->list[Path]:
    figure_dir=repo/"outputs/budgeted_risk/figures/stage0";figure_dir.mkdir(parents=True,exist_ok=True);paths=[]
    aggregated=results.groupby(["dataset","subject_id"],as_index=False).agg(j_context=("j_context","mean"),j_future=("j_future","mean"),raw_prediction=("raw_prediction","mean"))
    for dataset,current in aggregated.groupby("dataset"):
        for x,name,label in (("j_context","context_vs_future","J context"),("raw_prediction","predicted_vs_future","Predicted J")):
            path=figure_dir/f"{dataset}_{name}.png";fig,ax=plt.subplots(figsize=(5,4));ax.scatter(current[x],current.j_future,s=22,alpha=.75);ax.plot([0,20],[0,20],"--",color="gray",linewidth=1);ax.set(xlabel=label,ylabel="Future J",title=f"{dataset.upper()}: {label} vs Future J",xlim=(-.5,20.5),ylim=(-.5,20.5));fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig);paths.append(path)
    return paths


def build_delivery_manifest(repo:Path)->dict[str,Any]:
    files=[]
    for base in (repo/"delivery/budgeted_risk",repo/"outputs/budgeted_risk/stage0",repo/"outputs/budgeted_risk/figures/stage0"):
        if base.exists():
            files.extend(path for path in base.rglob("*") if path.is_file() and path.name!="DELIVERY_MANIFEST.json")
    payload={"schema_version":"budgeted-risk-delivery-v1","files":[{"path":str(path.relative_to(repo)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size} for path in sorted(set(files))]}
    atomic_json(payload,repo/"delivery/budgeted_risk/DELIVERY_MANIFEST.json");return payload

