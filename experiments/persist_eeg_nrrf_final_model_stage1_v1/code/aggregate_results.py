"""Aggregate locked NRRF-v1 Phase-A results and apply the preregistered gate."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
OUT, PROTOCOL = EXP / "outputs", EXP / "protocol"
BOOTSTRAPS = 10_000


def clean(value: Any) -> Any:
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def boot(values: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(0); draws = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return {"mean_pp": float(values.mean() * 100), "median_pp": float(np.median(values) * 100), "ci_low_pp": float(np.quantile(draws, .025) * 100), "ci_high_pp": float(np.quantile(draws, .975) * 100)}


def terminal(open_ok: bool, wbcic_ok: bool, robust_ok: bool) -> str:
    if open_ok and wbcic_ok and robust_ok: return "NRRF_STAGE1_SEED0_PASS_EXPAND"
    if open_ok and not wbcic_ok: return "NRRF_WBCIC_REPAIR_FAIL_STOP"
    if wbcic_ok and not open_ok: return "NRRF_OPENBMI_UPSIDE_NOT_PRESERVED_STOP"
    if not robust_ok: return "NRRF_ROBUST_OBJECTIVE_NOT_JUSTIFIED_STOP"
    return "NRRF_CONSTRUCTIVE_FAIL_STOP"


def main() -> int:
    subject = pd.read_csv(OUT / "PHASE_A_SUBJECT_RESULTS.csv")
    if len(subject) == 0 or not subject.primary.all(): raise RuntimeError("primary Phase-A subject result table missing")
    required = {(d, f, m) for d in ("OpenBMI", "WBCIC") for f in range(5) for m in ("EEGNet", "LiteBN", "FROZEN-LOGIT50", "JOINT-CE", "NRRF-v1")}
    actual = set(zip(subject.dataset, subject.fold, subject.method))
    if required - actual: raise RuntimeError(f"incomplete Phase-A primary outcomes: {sorted(required-actual)}")
    wide = subject.pivot(index=["dataset", "fold", "subject_id"], columns="method", values=["BA", "macro_F1", "accuracy"]).reset_index()
    wide.columns = ["_".join(str(x) for x in col if str(x)) for col in wide.columns]
    for method in ("LiteBN", "FROZEN-LOGIT50", "JOINT-CE", "NRRF-v1"):
        wide[f"delta_{method}_vs_EEGNet_pp"] = (wide[f"BA_{method}"] - wide["BA_EEGNet"]) * 100
        wide[f"delta_{method}_vs_LOGIT50_pp"] = (wide[f"BA_{method}"] - wide["BA_FROZEN-LOGIT50"]) * 100
    wide.to_csv(OUT / "PHASE_A_SUBJECT_RESULTS.csv", index=False)
    fold_rows = []
    for (dataset, fold), group in wide.groupby(["dataset", "fold"]):
        base = group.BA_EEGNet.mean()
        for method in ("EEGNet", "LiteBN", "FROZEN-LOGIT50", "JOINT-CE", "NRRF-v1"):
            delta = (group[f"BA_{method}"] - group.BA_EEGNet).to_numpy()
            fold_rows.append({"dataset":dataset,"fold":int(fold),"seed":0,"method":method,"mean_subject_BA":float(group[f"BA_{method}"].mean()),"mean_subject_macro_F1":float(group[f"macro_F1_{method}"].mean()),"mean_subject_accuracy":float(group[f"accuracy_{method}"].mean()),"delta_vs_EEGNet_pp":float((group[f"BA_{method}"].mean()-base)*100),"median_subject_delta_vs_EEGNet_pp":float(np.median(delta)*100),"harm_le_minus_1pp_fraction":float((delta<=-.01).mean()),"harm_le_minus_3pp_fraction":float((delta<=-.03).mean()),"gain_ge_1pp_fraction":float((delta>=.01).mean()),"worst_subject_delta_pp":float(delta.min()*100),"n_subjects":len(group)})
    folds = pd.DataFrame(fold_rows); folds.to_csv(OUT / "PHASE_A_FOLD_RESULTS.csv", index=False)
    dataset_rows = []; checks = {}
    for dataset in ("OpenBMI", "WBCIC"):
        group, f = wide[wide.dataset==dataset], folds[(folds.dataset==dataset)&(folds.method=="NRRF-v1")]
        delta = (group["BA_NRRF-v1"]-group.BA_EEGNet).to_numpy(); joint = (group["BA_JOINT-CE"]-group.BA_EEGNet).to_numpy()
        b, jb = boot(delta), boot(joint)
        robust_ba = b["mean_pp"] - jb["mean_pp"]
        robust_harm_reduction = float((joint<=-.01).mean()-(delta<=-.01).mean())*100
        row = {"dataset":dataset,"EEGNet_BA":float(group.BA_EEGNet.mean()),"LiteBN_BA":float(group.BA_LiteBN.mean()),"Frozen_LOGIT50_BA":float(group["BA_FROZEN-LOGIT50"].mean()),"JOINT_CE_BA":float(group["BA_JOINT-CE"].mean()),"NRRF_BA":float(group["BA_NRRF-v1"].mean()),"NRRF_gain_vs_EEGNet_pp":b["mean_pp"],"NRRF_median_subject_gain_pp":b["median_pp"],"NRRF_bootstrap_ci_low_pp":b["ci_low_pp"],"NRRF_bootstrap_ci_high_pp":b["ci_high_pp"],"NRRF_gain_vs_LOGIT50_pp":float((group["BA_NRRF-v1"]-group["BA_FROZEN-LOGIT50"]).mean()*100),"NRRF_harm_le_minus_1pp_fraction":float((delta<=-.01).mean()),"NRRF_harm_le_minus_3pp_fraction":float((delta<=-.03).mean()),"NRRF_worst_subject_delta_pp":float(delta.min()*100),"NRRF_positive_folds":int((f.delta_vs_EEGNet_pp>0).sum()),"NRRF_worst_fold_delta_pp":float(f.delta_vs_EEGNet_pp.min()),"NRRF_vs_JOINT_CE_meanBA_pp":robust_ba,"WBCIC_harm_reduction_vs_JOINT_CE_pp":robust_harm_reduction}
        dataset_rows.append(row); checks[dataset] = row
    ds = pd.DataFrame(dataset_rows); ds.to_csv(OUT / "PHASE_A_DATASET_RESULTS.csv", index=False)
    o,w = checks["OpenBMI"], checks["WBCIC"]
    open_ok = o["NRRF_gain_vs_EEGNet_pp"] >= 3 and o["NRRF_positive_folds"] == 5 and o["NRRF_worst_fold_delta_pp"] > -.5 and o["NRRF_harm_le_minus_1pp_fraction"] <= .10
    wbcic_ok = w["NRRF_gain_vs_EEGNet_pp"] >= 1 and w["NRRF_positive_folds"] >= 4 and w["NRRF_worst_fold_delta_pp"] > -2 and w["NRRF_median_subject_gain_pp"] > 0 and w["NRRF_harm_le_minus_1pp_fraction"] < .15
    robust_ok = w["NRRF_vs_JOINT_CE_meanBA_pp"] >= .30 or w["WBCIC_harm_reduction_vs_JOINT_CE_pp"] >= 5
    value = terminal(open_ok, wbcic_ok, robust_ok)
    failure = {"OpenBMI": {"mean_gain_ge_3pp": o["NRRF_gain_vs_EEGNet_pp"] >= 3, "five_positive_folds":o["NRRF_positive_folds"]==5,"no_fold_le_minus_0_5pp":o["NRRF_worst_fold_delta_pp"]>-.5,"harm_le_10pct":o["NRRF_harm_le_minus_1pp_fraction"]<=.10},"WBCIC":{"mean_gain_ge_1pp":w["NRRF_gain_vs_EEGNet_pp"]>=1,"at_least_four_positive_folds":w["NRRF_positive_folds"]>=4,"no_fold_le_minus_2pp":w["NRRF_worst_fold_delta_pp"]>-2,"positive_median_subject_delta":w["NRRF_median_subject_gain_pp"]>0,"harm_lt_15pct":w["NRRF_harm_le_minus_1pp_fraction"]<.15},"robust_objective":{"mean_BA_ge_joint_by_0_30pp":w["NRRF_vs_JOINT_CE_meanBA_pp"]>=.30,"harm_reduction_ge_5pp":w["WBCIC_harm_reduction_vs_JOINT_CE_pp"]>=5}}
    write_json(OUT / "PHASE_A_GATE.json", {"terminal":value,"openbmi_pass":open_ok,"wbcic_pass":wbcic_ok,"robust_objective_pass":robust_ok,"checks":failure})
    lines=["# NRRF-v1 Phase-A Decision","","| Metric | OpenBMI | WBCIC |","|---|---:|---:|"]
    for label,key in [("EEGNet BA","EEGNet_BA"),("LiteBN BA","LiteBN_BA"),("Frozen LOGIT50 BA","Frozen_LOGIT50_BA"),("JOINT-CE BA","JOINT_CE_BA"),("NRRF BA","NRRF_BA"),("NRRF gain vs EEGNet (pp)","NRRF_gain_vs_EEGNet_pp"),("NRRF gain vs LOGIT50 (pp)","NRRF_gain_vs_LOGIT50_pp"),("Median subject gain (pp)","NRRF_median_subject_gain_pp"),("Bootstrap CI (pp)",None),("Positive folds","NRRF_positive_folds"),("Harm ≤ −1 pp","NRRF_harm_le_minus_1pp_fraction")]:
        vals=[]
        for row in (o,w): vals.append(f"[{row['NRRF_bootstrap_ci_low_pp']:+.3f}, {row['NRRF_bootstrap_ci_high_pp']:+.3f}]" if key is None else (f"{row[key]:.3f}" if isinstance(row[key],float) else str(row[key])))
        lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
    lines += ["",f"WBCIC NRRF − JOINT-CE mean BA: {w['NRRF_vs_JOINT_CE_meanBA_pp']:+.3f} pp.",f"WBCIC harm reduction versus JOINT-CE: {w['WBCIC_harm_reduction_vs_JOINT_CE_pp']:+.3f} pp.","",f"## Terminal\n\n**{value}**","", "No coefficient, alpha, architecture, batch rule, or epoch budget was changed after outer-development outcomes were produced."]
    (OUT / "PHASE_A_DECISION.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(value, flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
