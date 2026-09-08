"""Subject-unit aggregation and fixed decision rules for final confirmation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from load_final_carriers import OUTPUTS, PROTOCOL, REPO, clean, write_json

BOOTSTRAPS = 10_000


def bootstrap(delta: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    draws = rng.choice(delta, size=(BOOTSTRAPS, len(delta)), replace=True).mean(axis=1)
    return {"bootstrap_unit": "heldout_subject_after_15_replicate_mean", "bootstrap_seed": 0, "resamples": BOOTSTRAPS, "n_subjects": int(len(delta)), "mean_delta_pp": float(delta.mean() * 100), "median_delta_pp": float(np.median(delta) * 100), "ci_low_pp": float(np.quantile(draws, .025) * 100), "ci_high_pp": float(np.quantile(draws, .975) * 100)}


def _subject_average(replicates: pd.DataFrame) -> pd.DataFrame:
    mean = replicates.groupby(["dataset", "subject_id", "method"], as_index=False)[["BA", "macro_F1", "accuracy"]].mean()
    expected = {"OpenBMI": 14, "WBCIC": 10}
    counts = mean.groupby(["dataset", "subject_id"]).size()
    if not (counts == 3).all() or mean.groupby("dataset").subject_id.nunique().to_dict() != expected:
        raise RuntimeError("subject-method cardinality invalid")
    wide = mean.pivot(index=["dataset", "subject_id"], columns="method", values=["BA", "macro_F1", "accuracy"])
    wide.columns = [f"{method}_{metric}" for metric, method in wide.columns]
    wide = wide.reset_index()
    wide["LOGIT50_minus_EEGNet_pp"] = (wide["LOGIT50_BA"] - wide["EEGNet_BA"]) * 100
    wide["LOGIT50_minus_LiteBN_pp"] = (wide["LOGIT50_BA"] - wide["LiteBN_BA"]) * 100
    return wide.rename(columns={"EEGNet_macro_F1": "EEGNet_macroF1", "LiteBN_macro_F1": "LiteBN_macroF1", "LOGIT50_macro_F1": "LOGIT50_macroF1"})


def _replicate_stability(replicates: pd.DataFrame) -> pd.DataFrame:
    rep = replicates.groupby(["dataset", "fold", "seed", "method"], as_index=False)["BA"].mean()
    wide = rep.pivot(index=["dataset", "fold", "seed"], columns="method", values="BA").reset_index()
    if len(wide) != 30 or set(wide.columns) != {"dataset", "fold", "seed", "EEGNet", "LiteBN", "LOGIT50"}:
        raise RuntimeError("replicate grid invalid")
    wide["LOGIT50_minus_EEGNet_pp"] = (wide.LOGIT50 - wide.EEGNet) * 100
    fold_delta = wide.groupby(["dataset", "fold"])["LOGIT50_minus_EEGNet_pp"].mean().to_dict()
    seed_delta = wide.groupby(["dataset", "seed"])["LOGIT50_minus_EEGNet_pp"].mean().to_dict()
    wide["fold_mean_delta_pp"] = [float(fold_delta[(r.dataset, r.fold)]) for r in wide.itertuples()]
    wide["seed_mean_delta_pp"] = [float(seed_delta[(r.dataset, r.seed)]) for r in wide.itertuples()]
    return wide.sort_values(["dataset", "fold", "seed"]).reset_index(drop=True)


def _dataset_rows(subjects: pd.DataFrame, stability: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows, details = [], {}
    for dataset, frame in subjects.groupby("dataset", sort=False):
        delta = frame.LOGIT50_minus_EEGNet_pp.to_numpy() / 100
        bs = bootstrap(delta)
        harm5 = float((delta <= -.05).mean())
        no_collapse = bool(bs["mean_delta_pp"] > -5.0)
        open_strong = dataset == "OpenBMI" and bs["mean_delta_pp"] >= 2 and bs["median_delta_pp"] > 0 and bs["ci_low_pp"] > 0
        wbcic_strong = dataset == "WBCIC" and bs["mean_delta_pp"] >= .5 and bs["median_delta_pp"] > 0 and int((delta > 0).sum()) >= 6 and no_collapse
        status = "STRONG" if (open_strong or wbcic_strong) else ("POSITIVE" if bs["mean_delta_pp"] > 0 and (dataset != "WBCIC" or bs["median_delta_pp"] >= 0) else "NOT CONFIRMED")
        rep = stability[stability.dataset == dataset]
        rows.append({"dataset": dataset, "EEGNet_BA": float(frame.EEGNet_BA.mean()), "LiteBN_BA": float(frame.LiteBN_BA.mean()), "FROZEN_LOGIT50_BA": float(frame.LOGIT50_BA.mean()), "LOGIT50_minus_EEGNet_pp": bs["mean_delta_pp"], "median_delta_pp": bs["median_delta_pp"], "bootstrap_ci_low_pp": bs["ci_low_pp"], "bootstrap_ci_high_pp": bs["ci_high_pp"], "positive_subjects": int((delta > 0).sum()), "harmed_subjects": int((delta < 0).sum()), "tied_subjects": int((delta == 0).sum()), "LOGIT50_minus_LiteBN_pp": float((frame.LOGIT50_BA.mean() - frame.LiteBN_BA.mean()) * 100), "LOGIT50_minus_max_global_carrier_pp": float((frame.LOGIT50_BA.mean() - max(frame.EEGNet_BA.mean(), frame.LiteBN_BA.mean())) * 100), "n_subjects": int(len(frame)), "replicate_mean_delta_pp": float(rep.LOGIT50_minus_EEGNet_pp.mean()), "replicate_min_delta_pp": float(rep.LOGIT50_minus_EEGNet_pp.min()), "replicate_max_delta_pp": float(rep.LOGIT50_minus_EEGNet_pp.max()), "positive_replicates": int((rep.LOGIT50_minus_EEGNet_pp > 0).sum()), "no_catastrophic_dataset_collapse": no_collapse, "harm_le_minus_5pp_fraction": harm5, "confirmation": status})
        details[dataset] = {"status": status, "bootstrap": bs, "subject_delta_pp": delta * 100, "no_catastrophic_dataset_collapse": no_collapse}
    return pd.DataFrame(rows), details


def _harm_profile(subjects: pd.DataFrame) -> dict[str, Any]:
    report = {}
    for dataset, frame in subjects.groupby("dataset", sort=False):
        d = frame.LOGIT50_minus_EEGNet_pp
        ordered = frame.sort_values("LOGIT50_minus_EEGNet_pp")
        report[dataset] = {"relative_to": "EEGNet", "n_subjects": int(len(frame)), "positive": int((d > 0).sum()), "positive_fraction": float((d > 0).mean()), "gain_ge_plus_1pp": int((d >= 1).sum()), "gain_ge_plus_3pp": int((d >= 3).sum()), "abs_delta_lt_0_5pp": int((d.abs() < .5).sum()), "harm_le_minus_1pp": int((d <= -1).sum()), "harm_le_minus_3pp": int((d <= -3).sum()), "harm_le_minus_5pp": int((d <= -5).sum()), "largest_improvement": {"subject_id": str(ordered.iloc[-1].subject_id), "delta_pp": float(ordered.iloc[-1].LOGIT50_minus_EEGNet_pp)}, "largest_harm": {"subject_id": str(ordered.iloc[0].subject_id), "delta_pp": float(ordered.iloc[0].LOGIT50_minus_EEGNet_pp)}, "catastrophic_subject_harm_pattern": bool((d <= -5).mean() >= .30), "catastrophic_definition": ">=30% held-out subjects with LOGIT50 minus EEGNet <= -5 pp"}
    return report


def _development_vs_holdout(dataset_rows: pd.DataFrame) -> pd.DataFrame:
    source = REPO / "experiments" / "persist_eeg_frozen_fusion_headroom_v1" / "outputs" / "DATASET_AGGREGATE_RESULTS.csv"
    frame = pd.read_csv(source)
    development = frame[frame.method == "LOGIT50"].set_index("dataset").mean_delta_pp.to_dict()
    out = []
    for row in dataset_rows.itertuples():
        out.append({"dataset": row.dataset, "development_LOGIT50_minus_EEGNet_pp": float(development[row.dataset]), "heldout_LOGIT50_minus_EEGNet_pp": float(row.LOGIT50_minus_EEGNet_pp), "development_source": str(source)})
    return pd.DataFrame(out)


def _terminal(rows: pd.DataFrame) -> str:
    data = rows.set_index("dataset")
    o, w = data.loc["OpenBMI"], data.loc["WBCIC"]
    if o.confirmation == "STRONG" and w.confirmation == "STRONG": return "FINAL_DUALDATASET_STRONG_CONFIRMATION"
    if o.LOGIT50_minus_EEGNet_pp > 0 and w.LOGIT50_minus_EEGNet_pp > 0: return "FINAL_DUALDATASET_POSITIVE_CONFIRMATION"
    if o.LOGIT50_minus_EEGNet_pp > 0 and w.LOGIT50_minus_EEGNet_pp <= 0: return "FINAL_OPENBMI_ONLY_CONFIRMATION"
    if w.LOGIT50_minus_EEGNet_pp > 0 and o.LOGIT50_minus_EEGNet_pp <= 0: return "FINAL_WBCIC_ONLY_CONFIRMATION"
    return "FINAL_HELDOUT_CONFIRMATION_FAIL"


def _decision_markdown(rows: pd.DataFrame, dev: pd.DataFrame, comp: dict[str, Any], harm: dict[str, Any], terminal: str) -> str:
    r = rows.set_index("dataset")
    table = ["| Dataset | EEGNet BA | LiteBN BA | Frozen LOGIT50 BA | Δ vs EEGNet | Median Δ | 95% CI | Positive subjects |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("OpenBMI", "WBCIC"):
        x = r.loc[dataset]
        table.append(f"| {dataset} | {x.EEGNet_BA:.4f} | {x.LiteBN_BA:.4f} | {x.FROZEN_LOGIT50_BA:.4f} | {x.LOGIT50_minus_EEGNet_pp:+.3f} pp | {x.median_delta_pp:+.3f} pp | [{x.bootstrap_ci_low_pp:+.3f}, {x.bootstrap_ci_high_pp:+.3f}] pp | {int(x.positive_subjects)}/{int(x.n_subjects)} |")
    d = dev.set_index("dataset")
    comp_status = "YES" if all(comp[x]["eegnet_only_correct_fraction"] > 0 and comp[x]["litebn_only_correct_fraction"] > 0 for x in ("OpenBMI", "WBCIC")) else "NO"
    lite_both = "YES" if all(r.loc[x].LOGIT50_minus_LiteBN_pp > 0 for x in ("OpenBMI", "WBCIC")) else "NO"
    harm_yes = "YES" if any(harm[x]["catastrophic_subject_harm_pattern"] for x in harm) else "NO"
    return "\n".join(["# PERSIST-EEG Final Held-Out Subject Confirmation", "", "Final predictor was frozen before held-out evaluation: `FROZEN_LOGIT50`.", "", *table, "", "## Development vs held-out", "", "| Dataset | Development Δ vs EEGNet | Held-out Δ vs EEGNet |", "|---|---:|---:|", *[f"| {x} | {d.loc[x].development_LOGIT50_minus_EEGNet_pp:+.3f} pp | {d.loc[x].heldout_LOGIT50_minus_EEGNet_pp:+.3f} pp |" for x in ("OpenBMI", "WBCIC")], "", f"1. OpenBMI held-out confirmation: **{r.loc['OpenBMI'].confirmation}**.", f"2. WBCIC held-out confirmation: **{r.loc['WBCIC'].confirmation}**.", f"3. Does the final fusion outperform EEGNet on both held-out datasets? **{'YES' if all(r.loc[x].LOGIT50_minus_EEGNet_pp > 0 for x in ('OpenBMI','WBCIC')) else 'NO'}**.", f"4. Does the final fusion outperform LiteBN on both held-out datasets? **{lite_both}**.", f"5. Does carrier complementarity remain observable? **{comp_status}**.", f"6. Is there any catastrophic subject-level harm pattern? **{harm_yes}**.", f"7. Overall terminal: `{terminal}`.", "", "8. Final scientific interpretation: The primary unit is the held-out subject after averaging its 15 fixed carrier-pair replicates. The predictor, checkpoints, source-only normalizers, and 50/50 logit rule were fixed before held-out labels were opened. The terminal follows the preregistered dataset gates without subject, seed, or fold exclusion. This result is final confirmation evidence and does not license model retuning.", ""])


def aggregate() -> None:
    path = OUTPUTS / "REPLICATE_SUBJECT_RESULTS.csv"
    replicates = pd.read_csv(path)
    if len(replicates) != 1080: raise RuntimeError("final replicate result row count must be 1080")
    subject = _subject_average(replicates)
    subject = subject[["dataset", "subject_id", "EEGNet_BA", "LiteBN_BA", "LOGIT50_BA", "LOGIT50_minus_EEGNet_pp", "LOGIT50_minus_LiteBN_pp", "EEGNet_macroF1", "LiteBN_macroF1", "LOGIT50_macroF1", "EEGNet_accuracy", "LiteBN_accuracy", "LOGIT50_accuracy"]]
    stability = _replicate_stability(replicates)
    dataset_rows, detail = _dataset_rows(subject, stability)
    harm = _harm_profile(subject)
    comp = json.loads((OUTPUTS / "COMPLEMENTARITY.json").read_text(encoding="utf-8"))
    dev = _development_vs_holdout(dataset_rows)
    terminal = _terminal(dataset_rows)
    subject.to_csv(OUTPUTS / "SUBJECT_AVERAGED_RESULTS.csv", index=False)
    stability.to_csv(OUTPUTS / "REPLICATE_STABILITY.csv", index=False)
    dataset_rows.to_csv(OUTPUTS / "DATASET_RESULTS.csv", index=False)
    dev.to_csv(OUTPUTS / "DEVELOPMENT_VS_HOLDOUT.csv", index=False)
    write_json(OUTPUTS / "SUBJECT_HARM_PROFILE.json", harm)
    write_json(OUTPUTS / "FINAL_TERMINAL.json", {"terminal": terminal, "openbmi_confirmation": detail["OpenBMI"]["status"], "wbcic_confirmation": detail["WBCIC"]["status"], "dataset_rows": dataset_rows.to_dict("records"), "subject_is_primary_unit": True, "replicates_per_subject": 15})
    (OUTPUTS / "FINAL_DECISION.md").write_text(_decision_markdown(dataset_rows, dev, comp, harm, terminal), encoding="utf-8")


if __name__ == "__main__":
    aggregate()
