"""Frozen EEGNet + LiteBN audit: inference only, no model updates.

The command is deliberately fail-closed.  It verifies the archived selected
checkpoint hashes and reproduces the source fold/seed BA before reporting a
single fusion result.  It never constructs an optimizer, invokes backward, or
loads a subject outside the copied FIVEFOLD_SPLIT search lists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


BOOTSTRAPS = 10_000
METHODS = ("EEGNet", "LiteBN", "LOGIT50", "PROB50")
VARIANTS = (("selected_best", "selected_best.pt"), ("epoch60", "epoch60.pt"))
FROZEN_WEIGHTS = {"LOGIT50": [0.5, 0.5], "PROB50": [0.5, 0.5]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "BA": float(balanced_accuracy_score(y, prediction)),
        "macro_F1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
    }


def bootstrap_pp(delta: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    draws = rng.choice(delta, size=(BOOTSTRAPS, len(delta)), replace=True).mean(axis=1)
    return {
        "bootstrap_unit": "subject_after_seed_mean",
        "bootstrap_seed": 0,
        "resamples": BOOTSTRAPS,
        "n_subjects": int(len(delta)),
        "mean_delta_pp": float(delta.mean() * 100),
        "median_delta_pp": float(np.median(delta) * 100),
        "ci_low_pp": float(np.quantile(draws, 0.025) * 100),
        "ci_high_pp": float(np.quantile(draws, 0.975) * 100),
        "improved_subjects": int((delta > 0).sum()),
        "harmed_subjects": int((delta < 0).sum()),
        "tied_subjects": int((delta == 0).sum()),
    }


def module(name: str, channels: int, carrier: Any, EEGNet: Any) -> torch.nn.Module:
    if name == "EEGNet":
        return EEGNet(channels)
    if name == "LiteBN":
        return carrier.CompactLite(channels, "bn")
    raise ValueError(name)


def load_model(name: str, channels: int, path: Path, device: torch.device, carrier: Any, EEGNet: Any) -> torch.nn.Module:
    model = module(name, channels, carrier, EEGNet).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    incompat = model.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"strict state mismatch: {path}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError(f"frozen/eval invariant failed: {path}")
    return model


def batched_logits(model: torch.nn.Module, indices: np.ndarray, cache: Any) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            x, _ = cache.batch(indices[start : start + 128])
            chunks.append(model(x)[0].float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def expected_ba(source_fold: pd.DataFrame, dataset: str, fold: int, seed: int, model_name: str) -> float:
    row = source_fold[(source_fold.dataset == dataset) & (source_fold.fold == fold) & (source_fold.seed == seed)]
    if len(row) != 1:
        raise RuntimeError(f"source result cell missing or ambiguous: {dataset} fold={fold} seed={seed}")
    return float(row.iloc[0][f"{model_name}_BA"])


def primary_subject_frame(rows: pd.DataFrame) -> pd.DataFrame:
    selected = rows[(rows.checkpoint_variant == "selected_best") & rows.method.isin(METHODS)].copy()
    mean = selected.groupby(["dataset", "fold", "subject_id", "method"], as_index=False)[["BA", "macro_F1", "accuracy"]].mean()
    wide = mean.pivot(index=["dataset", "fold", "subject_id"], columns="method", values=["BA", "macro_F1", "accuracy"])
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    wide["delta_LiteBN_pp"] = (wide.BA_LiteBN - wide.BA_EEGNet) * 100
    wide["delta_LOGIT50_pp"] = (wide.BA_LOGIT50 - wide.BA_EEGNet) * 100
    wide["delta_PROB50_pp"] = (wide.BA_PROB50 - wide.BA_EEGNet) * 100
    wide["LOGIT50_minus_LiteBN_pp"] = (wide.BA_LOGIT50 - wide.BA_LiteBN) * 100
    wide["BA_SUBJECT_CARRIER_ORACLE"] = wide[["BA_EEGNet", "BA_LiteBN"]].max(axis=1)
    wide["delta_ORACLE_pp"] = (wide.BA_SUBJECT_CARRIER_ORACLE - wide.BA_EEGNet) * 100
    return wide


def classify_preference(row: pd.Series) -> str:
    lite = row["BA_LiteBN"] if "BA_LiteBN" in row.index else row["LiteBN"]
    eeg = row["BA_EEGNet"] if "BA_EEGNet" in row.index else row["EEGNet"]
    if lite > eeg + 0.005:
        return "LITEBN"
    if eeg > lite + 0.005:
        return "EEGNET"
    return "TIE"


def terminal_from(primary: pd.DataFrame, fold_seed: pd.DataFrame, comp: pd.DataFrame, preference: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    by_data = {name: group for name, group in primary.groupby("dataset")}
    logit = {name: group[group.method == "LOGIT50"].iloc[0] for name, group in by_data.items()}
    a = (
        logit["OpenBMI"].mean_delta_pp >= 2.0
        and int(logit["OpenBMI"].positive_folds) >= 4
        and int(logit["OpenBMI"].positive_seed_means) == 3
        and logit["OpenBMI"].median_delta_pp > 0
        and logit["WBCIC"].mean_delta_pp >= 1.0
        and int(logit["WBCIC"].positive_folds) >= 4
        and int(logit["WBCIC"].positive_seed_means) == 3
        and logit["WBCIC"].median_delta_pp > 0
        and logit["WBCIC"].harm_le_minus_1pp_fraction < 0.20
    )
    # These numeric interpretations are fixed in this source code before any
    # outcome is read: broad=at least 3 subjects; substantial=5% of trials;
    # stable=at least half of WBCIC subjects have a same-carrier 2/3+ signal.
    w = by_data["WBCIC"]
    o = by_data["OpenBMI"]
    oracle = w[w.method == "SUBJECT_CARRIER_ORACLE"].iloc[0]
    wb_best = float(w[w.method.isin(["EEGNet", "LiteBN"])].mean_BA.max())
    open_oracle = float(o[o.method == "SUBJECT_CARRIER_ORACLE"].mean_BA.iloc[0])
    open_lite = float(o[o.method == "LiteBN"].mean_BA.iloc[0])
    gain_subjects = int((preference[(preference.dataset == "WBCIC")].oracle_gain_over_best_global_pp >= 1).sum())
    stable = preference[(preference.dataset == "WBCIC")].stable_2of3_or_more.mean() >= 0.50
    wcomp = comp[(comp.checkpoint_variant == "selected_best") & (comp.dataset == "WBCIC")]
    e_only = float(wcomp.eegnet_only_fraction.mean())
    l_only = float(wcomp.litebn_only_fraction.mean())
    strong = (
        (float(oracle.mean_BA) - float(w[w.method == "EEGNet"].mean_BA.iloc[0])) * 100 >= 2.0
        and gain_subjects >= 3 and stable and e_only >= 0.05 and l_only >= 0.05
        and open_oracle >= open_lite
        and (float(oracle.mean_BA) - float(w[w.method == "LOGIT50"].mean_BA.iloc[0])) * 100 >= 1.0
    )
    limited = float(oracle.mean_BA) > wb_best + 0.003
    details = {"fixed_fusion_crossdataset_promising": bool(a), "strong_conditional_checks": {
        "oracle_over_EEGNet_pp": (float(oracle.mean_BA) - float(w[w.method == "EEGNet"].mean_BA.iloc[0])) * 100,
        "broad_subjects_ge_1pp": gain_subjects, "stable_preference": bool(stable),
        "eegnet_only_fraction": e_only, "litebn_only_fraction": l_only,
        "open_oracle_ge_litebn": bool(open_oracle >= open_lite),
        "wbcic_oracle_minus_logit_pp": (float(oracle.mean_BA) - float(w[w.method == "LOGIT50"].mean_BA.iloc[0])) * 100,
    }}
    if a:
        return "FIXED_FUSION_CROSSDATASET_PROMISING", details
    if strong:
        return "STRONG_CONDITIONAL_HEADROOM", details
    if limited:
        return "LIMITED_CONDITIONAL_HEADROOM", details
    return "NO_USEFUL_COMPLEMENTARITY", details


def make_outputs(subject_rows: pd.DataFrame, comp: pd.DataFrame, source_fold: pd.DataFrame, outputs: Path, protocols: Path) -> str:
    selected = subject_rows[subject_rows.checkpoint_variant == "selected_best"].copy()
    fold_seed_rows: list[dict[str, Any]] = []
    for (dataset, fold, seed, method), group in selected.groupby(["dataset", "fold", "seed", "method"]):
        eeg = selected[(selected.dataset == dataset) & (selected.fold == fold) & (selected.seed == seed) & (selected.method == "EEGNet")].BA.mean()
        fold_seed_rows.append({"dataset": dataset, "fold": int(fold), "seed": int(seed), "checkpoint_variant": "selected_best", "method": method,
                               "mean_subject_BA": float(group.BA.mean()), "mean_subject_macro_F1": float(group.macro_F1.mean()),
                               "mean_subject_accuracy": float(group.accuracy.mean()), "delta_vs_EEGNet_pp": float((group.BA.mean() - eeg) * 100), "n_subjects": int(len(group))})
    fold_seed = pd.DataFrame(fold_seed_rows)
    # Reproduction is deliberately checked before a fusion outcome is emitted.
    reproduction = []
    for dataset, fold, seed in source_fold[["dataset", "fold", "seed"]].itertuples(index=False, name=None):
        for model in ("EEGNet", "LiteBN"):
            actual = float(fold_seed[(fold_seed.dataset == dataset) & (fold_seed.fold == fold) & (fold_seed.seed == seed) & (fold_seed.method == model)].mean_subject_BA.iloc[0])
            expected = expected_ba(source_fold, dataset, int(fold), int(seed), model)
            reproduction.append({"dataset": dataset, "fold": int(fold), "seed": int(seed), "model": model, "expected_BA": expected, "recomputed_BA": actual, "absolute_difference": abs(expected - actual), "within_1e-6": abs(expected - actual) <= 1e-6})
    reproduction_frame = pd.DataFrame(reproduction)
    write_json(protocols / "TESTS.json", {"exact_previous_fivefold_split_reused": True, "exact_previous_60_selected_checkpoints_used": True,
        "all_checkpoint_sha256_verified": True, "baseline_reproduction": reproduction, "baseline_reproduced_within_1e-6": bool(reproduction_frame["within_1e-6"].all()),
        "optimizer_instantiated": False, "backward_called": False, "all_model_parameters_frozen": True, "all_models_eval": True,
        "LOGIT50_weights": FROZEN_WEIGHTS["LOGIT50"], "PROB50_weights": FROZEN_WEIGHTS["PROB50"],
        "fusion_weight_search": False, "learned_gate": False, "subject_dependent_deployable_rule": False,
        "oracle_marked_non_deployable": True, "sealed_holdout_accessed": False})
    if not reproduction_frame["within_1e-6"].all():
        write_csv(fold_seed, outputs / "FOLD_SEED_FUSION_RESULTS.csv")
        (outputs / "DECISION.md").write_text("# Frozen EEGNet + LiteBN fusion/headroom audit\n\n`FROZEN_PREDICTION_REPRODUCTION_INVALID`\n", encoding="utf-8")
        raise RuntimeError("FROZEN_PREDICTION_REPRODUCTION_INVALID")
    # Keep an explicit non-deployable subject oracle in the same long table.
    oracle_rows = []
    for (dataset, fold, seed, subject), group in selected.groupby(["dataset", "fold", "seed", "subject_id"]):
        a, b = group.set_index("method").loc[["EEGNet", "LiteBN"], "BA"]
        oracle_rows.append({"dataset":dataset, "fold":int(fold), "seed":int(seed), "checkpoint_variant":"selected_best", "subject_id":subject,
                            "method":"SUBJECT_CARRIER_ORACLE", "trials":int(group.trials.iloc[0]), "BA":float(max(a,b)), "macro_F1":np.nan, "accuracy":np.nan,
                            "non_deployable":True})
    subject_seed = pd.concat([selected, pd.DataFrame(oracle_rows)], ignore_index=True)
    write_csv(subject_seed, outputs / "SUBJECT_SEED_FUSION_RESULTS.csv")
    write_csv(fold_seed, outputs / "FOLD_SEED_FUSION_RESULTS.csv")
    subject = primary_subject_frame(selected)
    write_csv(subject, outputs / "SUBJECT_AGGREGATE_RESULTS.csv")
    preference_seed = selected[selected.method.isin(["EEGNet", "LiteBN"])].pivot(index=["dataset", "fold", "subject_id", "seed"], columns="method", values="BA").reset_index()
    preference_seed["seed_preference"] = preference_seed.apply(classify_preference, axis=1)
    p_rows = []
    for (dataset, fold, sid), group in preference_seed.groupby(["dataset", "fold", "subject_id"]):
        srow = subject[(subject.dataset == dataset) & (subject.fold == fold) & (subject.subject_id == sid)].iloc[0]
        counts = group.seed_preference.value_counts().to_dict()
        p_rows.append({"dataset":dataset, "fold":int(fold), "subject_id":sid, "EEGNet_mean_BA":float(srow.BA_EEGNet), "LiteBN_mean_BA":float(srow.BA_LiteBN),
                       "preference":classify_preference(srow), "LiteBN_wins_3of3":int((group.seed_preference == "LITEBN").sum()) == 3,
                       "EEGNet_wins_3of3":int((group.seed_preference == "EEGNET").sum()) == 3,
                       "LiteBN_wins":int((group.seed_preference == "LITEBN").sum()), "EEGNet_wins":int((group.seed_preference == "EEGNET").sum()),
                       "ties":int((group.seed_preference == "TIE").sum()), "stable_2of3_or_more": bool(max(int((group.seed_preference == "LITEBN").sum()), int((group.seed_preference == "EEGNET").sum())) >= 2),
                       "oracle_gain_over_best_global_pp":float((srow.BA_SUBJECT_CARRIER_ORACLE - max(srow.BA_EEGNet, srow.BA_LiteBN)) * 100)})
    preference = pd.DataFrame(p_rows)
    # The oracle's reference is the best global carrier in its dataset, not the per-subject winner.
    for dataset, group in preference.groupby("dataset"):
        best_global = subject[subject.dataset == dataset][["BA_EEGNet", "BA_LiteBN"]].mean().max()
        preference.loc[group.index, "oracle_gain_over_best_global_pp"] = (subject.loc[subject.dataset == dataset, "BA_SUBJECT_CARRIER_ORACLE"].to_numpy() - best_global) * 100
    write_csv(preference, outputs / "CARRIER_PREFERENCE.csv")
    stability = {}
    for dataset, group in preference.groupby("dataset"):
        stability[dataset] = {"LiteBN_3of3":int((group.LiteBN_wins == 3).sum()), "LiteBN_2of3":int((group.LiteBN_wins == 2).sum()),
                              "tie_or_mixed":int((~((group.LiteBN_wins >= 2) | (group.EEGNet_wins >= 2))).sum()), "EEGNet_2of3":int((group.EEGNet_wins == 2).sum()), "EEGNet_3of3":int((group.EEGNet_wins == 3).sum())}
    write_json(outputs / "PREFERENCE_STABILITY.json", stability)
    dataset_rows = []
    for dataset, group in subject.groupby("dataset"):
        for method, column in (("EEGNet", "BA_EEGNet"), ("LiteBN", "BA_LiteBN"), ("LOGIT50", "BA_LOGIT50"), ("PROB50", "BA_PROB50"), ("SUBJECT_CARRIER_ORACLE", "BA_SUBJECT_CARRIER_ORACLE")):
            delta = group[column].to_numpy() - group.BA_EEGNet.to_numpy()
            bs = bootstrap_pp(delta)
            folds = fold_seed[(fold_seed.dataset == dataset) & (fold_seed.method == method)] if method in METHODS else pd.DataFrame()
            fold_means = folds.groupby("fold").delta_vs_EEGNet_pp.mean() if len(folds) else pd.Series(dtype=float)
            seed_means = folds.groupby("seed").delta_vs_EEGNet_pp.mean() if len(folds) else pd.Series(dtype=float)
            dataset_rows.append({"dataset":dataset, "method":method, "mean_BA":float(group[column].mean()), **bs,
                                 "positive_folds":int((fold_means > 0).sum()), "positive_seed_means":int((seed_means > 0).sum()),
                                 "harm_le_minus_1pp_fraction":float((delta <= -0.01).mean()), "harm_le_minus_3pp_fraction":float((delta <= -0.03).mean()),
                                 "harm_le_minus_5pp_fraction":float((delta <= -0.05).mean()), "gain_ge_1pp_fraction":float((delta >= .01).mean()), "negligible_abs_lt_0_5pp_fraction":float((np.abs(delta) < .005).mean())})
    aggregate = pd.DataFrame(dataset_rows)
    write_csv(aggregate, outputs / "DATASET_AGGREGATE_RESULTS.csv")
    comp_selected = comp[comp.checkpoint_variant == "selected_best"].copy()
    comp_selected["eegnet_only_fraction"] = comp_selected.eegnet_only_correct / comp_selected.trials
    comp_selected["litebn_only_fraction"] = comp_selected.litebn_only_correct / comp_selected.trials
    comp_selected["disagreement_rate"] = comp_selected.disagree / comp_selected.trials
    comp_selected["eegnet_win_fraction_among_disagreement"] = np.where(comp_selected.disagree > 0, comp_selected.eegnet_only_correct / comp_selected.disagree, np.nan)
    comp_selected["litebn_win_fraction_among_disagreement"] = np.where(comp_selected.disagree > 0, comp_selected.litebn_only_correct / comp_selected.disagree, np.nan)
    comp_selected["trial_complementarity_upper_BA"] = (comp_selected.oracle_class0 + comp_selected.oracle_class1) / 2
    write_csv(comp_selected, outputs / "COMPLEMENTARITY_RESULTS.csv")
    oracle_headroom = {}
    concentration = {}
    harm = {}
    for dataset, group in subject.groupby("dataset"):
        best_global_method = "LiteBN" if group.BA_LiteBN.mean() > group.BA_EEGNet.mean() else "EEGNet"
        best_global = group[["BA_EEGNet", "BA_LiteBN"]].mean().max()
        oracle_gain = group.BA_SUBJECT_CARRIER_ORACLE - best_global
        logit_ba = group.BA_LOGIT50.mean()
        oracle_headroom[dataset] = {"EEGNet_BA":float(group.BA_EEGNet.mean()), "LiteBN_BA":float(group.BA_LiteBN.mean()), "LOGIT50_BA":float(logit_ba), "PROB50_BA":float(group.BA_PROB50.mean()),
                                    "SUBJECT_CARRIER_ORACLE_BA_NON_DEPLOYABLE":float(group.BA_SUBJECT_CARRIER_ORACLE.mean()), "best_global_carrier":best_global_method,
                                    "oracle_minus_best_global_pp":float(oracle_gain.mean()*100), "oracle_minus_LOGIT50_pp":float((group.BA_SUBJECT_CARRIER_ORACLE.mean()-logit_ba)*100)}
        positive = np.maximum(oracle_gain.to_numpy(), 0)
        top = np.sort(positive)[-max(1, math.ceil(len(positive)/4)):].sum()
        concentration[dataset] = {"median_oracle_improvement_over_best_global_pp":float(np.median(oracle_gain)*100), "subjects_oracle_gain_ge_1pp":int((oracle_gain >= .01).sum()),
                                  "subjects_oracle_gain_ge_3pp":int((oracle_gain >= .03).sum()), "subjects_oracle_gain_ge_5pp":int((oracle_gain >= .05).sum()),
                                  "top_quartile_contribution_fraction":float(top / positive.sum()) if positive.sum() else 0.0}
        harm[dataset] = {method: {"delta_ge_plus_1pp_fraction":float((group[f"BA_{method}"]-group.BA_EEGNet >= .01).mean()), "abs_delta_lt_0_5pp_fraction":float((np.abs(group[f"BA_{method}"]-group.BA_EEGNet) < .005).mean()),
                                  "delta_le_minus_1pp_fraction":float((group[f"BA_{method}"]-group.BA_EEGNet <= -.01).mean()), "delta_le_minus_3pp_fraction":float((group[f"BA_{method}"]-group.BA_EEGNet <= -.03).mean()), "delta_le_minus_5pp_fraction":float((group[f"BA_{method}"]-group.BA_EEGNet <= -.05).mean())} for method in ("LiteBN", "LOGIT50", "PROB50")}
    write_json(outputs / "ORACLE_HEADROOM.json", {"NON_DEPLOYABLE": True, **oracle_headroom})
    write_json(outputs / "ORACLE_GAIN_CONCENTRATION.json", concentration)
    write_json(outputs / "HARM_PROFILE.json", harm)
    epoch = subject_rows[subject_rows.checkpoint_variant == "epoch60"]
    epoch_rows = []
    for dataset, group in epoch.groupby("dataset"):
        pivot = group.pivot_table(index=["fold", "subject_id", "seed"], columns="method", values="BA").reset_index()
        for method in ("LiteBN", "LOGIT50"):
            d = pivot[method] - pivot.EEGNet
            fs = d.groupby(pivot.fold).mean(); ss = d.groupby(pivot.seed).mean()
            epoch_rows.append({"dataset":dataset, "method":method, "mean_delta_vs_EEGNet_pp":float(d.mean()*100), "positive_folds":int((fs > 0).sum()), "positive_seed_means":int((ss > 0).sum())})
    write_json(outputs / "EPOCH60_SECONDARY.json", {"secondary_only": True, "rows": epoch_rows})
    terminal, terminal_details = terminal_from(aggregate, fold_seed, comp_selected, preference)
    lite_gain = aggregate[(aggregate.dataset == "OpenBMI") & (aggregate.method == "LiteBN")].mean_delta_pp.iloc[0]
    logit_gain = aggregate[(aggregate.dataset == "OpenBMI") & (aggregate.method == "LOGIT50")].mean_delta_pp.iloc[0]
    retention = float(logit_gain / lite_gain) if lite_gain != 0 else None
    lines = ["# Frozen EEGNet + LiteBN fusion/headroom audit", "", "Primary results use only hash-verified `selected_best.pt` models, V8_SEARCH subjects, and the frozen five-fold split. No model was trained, updated, calibrated, stacked, or gated.", "", "| Metric | OpenBMI | WBCIC |", "|---|---:|---:|"]
    table = {d: aggregate[aggregate.dataset == d].set_index("method") for d in ("OpenBMI", "WBCIC")}
    for label, method, col in (("EEGNet BA", "EEGNet", "mean_BA"), ("LiteBN BA", "LiteBN", "mean_BA"), ("LiteBN delta (pp)", "LiteBN", "mean_delta_pp"), ("LOGIT50 BA", "LOGIT50", "mean_BA"), ("LOGIT50 delta (pp)", "LOGIT50", "mean_delta_pp"), ("PROB50 BA", "PROB50", "mean_BA"), ("PROB50 delta (pp)", "PROB50", "mean_delta_pp"), ("Subject Oracle BA (non-deployable)", "SUBJECT_CARRIER_ORACLE", "mean_BA"), ("LOGIT50 median subject delta (pp)", "LOGIT50", "median_delta_pp"), ("LOGIT50 harm <= -1pp", "LOGIT50", "harm_le_minus_1pp_fraction"), ("LOGIT50 positive folds", "LOGIT50", "positive_folds")):
        fmt = (lambda x: f"{x:.4f}") if col == "mean_BA" else ((lambda x: f"{x:.1%}") if col == "harm_le_minus_1pp_fraction" else (lambda x: f"{x:+.3f}" if "pp" in col else str(int(x))))
        lines.append(f"| {label} | {fmt(table['OpenBMI'].loc[method, col])} | {fmt(table['WBCIC'].loc[method, col])} |")
    lines.append(f"| Oracle delta over best single carrier (pp) | {oracle_headroom['OpenBMI']['oracle_minus_best_global_pp']:+.3f} | {oracle_headroom['WBCIC']['oracle_minus_best_global_pp']:+.3f} |")
    lines += ["", f"OpenBMI LOGIT50 retention of LiteBN gain: {retention:.1%}." if retention is not None else "OpenBMI LOGIT50 retention is undefined because LiteBN gain is zero.", "", "## Terminal", "", f"`{terminal}`", "", "The subject carrier oracle and trial complementarity upper bound are explicitly non-deployable label-informed diagnostics. They are not fusion methods and do not establish a viable gate.", "", "## Fixed decision checks", "", "```json", json.dumps(clean(terminal_details), indent=2, sort_keys=True), "```"]
    (outputs / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--openbmi-cache", required=True)
    parser.add_argument("--wbcic-cache", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repo, runtime = Path(args.repo).resolve(), Path(args.runtime).resolve()
    exp = repo / "experiments" / "persist_eeg_frozen_fusion_headroom_v1"
    source = repo / "experiments" / "persist_eeg_carrier_5fold_multiseed_stability_v1"
    code = exp / "code"; protocols, outputs = exp / "protocol", exp / "outputs"
    protocols.mkdir(parents=True, exist_ok=True); outputs.mkdir(parents=True, exist_ok=True)
    # Roots must be present before importing run_stage1, whose module constants
    # intentionally capture these exact cache locations.
    os.environ["PERSIST_OPENBMI_CACHE"] = str(Path(args.openbmi_cache).resolve())
    os.environ["PERSIST_WBCIC_CACHE"] = str(Path(args.wbcic_cache).resolve())
    sys.path[:0] = [str(source / "code"), str(repo / "experiments" / "persist_eeg_carrier_dualdataset_screen_v1" / "code"), str(repo / "experiments" / "persist_eeg_r2eeg_stage1_v1" / "code")]
    import run_stage1 as v1  # noqa: PLC0415
    import run_carrier_screen as carrier  # noqa: PLC0415
    from eegnet_locked import EEGNet  # noqa: PLC0415
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bounded frozen inference runtime")
    split_path, logs_path, source_fold_path = source / "protocol" / "FIVEFOLD_SPLIT.json", source / "outputs" / "TRAINING_LOGS.json", source / "outputs" / "FOLD_SEED_RESULTS.csv"
    split, logs, source_fold = json.loads(split_path.read_text(encoding="utf-8")), json.loads(logs_path.read_text(encoding="utf-8")), pd.read_csv(source_fold_path)
    if split.get("protocol") != "CARRIER_5FOLD_MULTISEED_STABILITY_V1":
        raise RuntimeError("frozen source split protocol mismatch")
    logs_by_key = {(r["dataset"], int(r["fold"]), int(r["seed"]), r["model"]): r for r in logs}
    write_json(protocols / "SOURCE_EXPERIMENT.json", {"source_branch":"codex/persist-eeg-carrier-5fold-multiseed-stability-v1", "source_experiment":str(source), "split_path":str(split_path), "split_sha256":sha256(split_path), "training_log_path":str(logs_path), "training_log_sha256":sha256(logs_path), "fold_results_sha256":sha256(source_fold_path), "source_commit_expected":"f86cf0f75376bbc9aaf4209606454449eed0ddcb"})
    write_json(protocols / "DATA_PROVENANCE.json", {"datasets":{"OpenBMI":{"cache_root":str(Path(args.openbmi_cache).resolve()), "source_sessions":[1], "future_outer_dev_session":[2]}, "WBCIC":{"cache_root":str(Path(args.wbcic_cache).resolve()), "source_sessions":[0,1], "future_outer_dev_session":[2]}}, "allowed_search_subjects":split["search_subjects"], "normalizer":"exact source run_stage1.normalizer semantics"})
    write_json(protocols / "FUSION_PROTOCOL.json", {"primary":"LOGIT50", "LOGIT50":"argmax(0.5*logits_EEGNet + 0.5*logits_LiteBN)", "PROB50":"argmax(0.5*softmax(logits_EEGNet) + 0.5*softmax(logits_LiteBN))", "weights":FROZEN_WEIGHTS, "weight_search":False, "learned_gate":False, "oracle":"NON_DEPLOYABLE label-informed diagnostic"})
    write_json(protocols / "HOLDOUT_ISOLATION_AUDIT.json", {"V8_INTERNAL_HOLDOUT_loaded":False, "V8_INTERNAL_HOLDOUT_labels_loaded":False, "WBCIC_true_outer_loaded":False, "WBCIC_true_outer_labels_loaded":False, "loaded_scope":"only FIVEFOLD_SPLIT V8_SEARCH outer_dev subjects"})
    # Verify all primary/secondary archival files and strict state compatibility before any outcome evaluation.
    provenance: list[dict[str, Any]] = []
    for dataset, channels in (("OpenBMI", 62), ("WBCIC", 58)):
        for fold in range(5):
            for seed in range(3):
                for model_name in ("EEGNet", "LiteBN"):
                    log = logs_by_key[(dataset, fold, seed, model_name)]
                    base = runtime / f"{dataset.lower()}_fold{fold}_seed{seed}_{model_name.lower()}"
                    for variant, filename in VARIANTS:
                        path = base / filename
                        expected = log["checkpoint_sha256" if variant == "selected_best" else "epoch60_checkpoint_sha256"]
                        row = {"dataset":dataset,"fold":fold,"seed":seed,"model":model_name,"variant":variant,"path":str(path),"exists":path.is_file(),"expected_sha256":expected}
                        if path.is_file():
                            row["actual_sha256"] = sha256(path); row["sha_match"] = row["actual_sha256"] == expected
                            try:
                                checked = load_model(model_name, channels, path, device, carrier, EEGNet)
                                row["strict_load"] = True; row["all_frozen"] = not any(p.requires_grad for p in checked.parameters()); row["eval_mode"] = not checked.training
                                del checked
                            except Exception as error:  # provenance must show the actual cause
                                row["strict_load"] = False; row["load_error"] = repr(error); row["all_frozen"] = False; row["eval_mode"] = False
                        else:
                            row.update({"actual_sha256":None,"sha_match":False,"strict_load":False,"all_frozen":False,"eval_mode":False})
                        provenance.append(row)
    write_json(protocols / "CHECKPOINT_PROVENANCE.json", {"expected_primary_cells":60,"verified_rows":provenance,"primary_valid":bool(all(r["exists"] and r["sha_match"] and r["strict_load"] and r["all_frozen"] and r["eval_mode"] for r in provenance if r["variant"] == "selected_best")), "secondary_epoch60_verified":bool(all(r["exists"] and r["sha_match"] and r["strict_load"] for r in provenance if r["variant"] == "epoch60"))})
    if not all(r["exists"] and r["sha_match"] and r["strict_load"] and r["all_frozen"] and r["eval_mode"] for r in provenance if r["variant"] == "selected_best"):
        (outputs / "DECISION.md").write_text("# Frozen EEGNet + LiteBN fusion/headroom audit\n\n`FROZEN_FUSION_CHECKPOINT_INVALID`\n", encoding="utf-8")
        raise RuntimeError("FROZEN_FUSION_CHECKPOINT_INVALID")
    all_rows: list[dict[str, Any]] = []; comp_rows: list[dict[str, Any]] = []
    for dataset in ("OpenBMI", "WBCIC"):
        bundle = v1.load_bundle(dataset, split["search_subjects"][dataset])
        for fold_data in split["folds"][dataset]:
            fold = int(fold_data["fold_id"])
            mean, std, norm = v1.normalizer(bundle, fold_data["inner_train_subjects"])
            cache = carrier.GPUCache(bundle, mean, std, device)
            for seed in range(3):
                for variant, filename in VARIANTS:
                    loaded = {name: load_model(name, bundle.channels, runtime / f"{dataset.lower()}_fold{fold}_seed{seed}_{name.lower()}" / filename, device, carrier, EEGNet) for name in ("EEGNet", "LiteBN")}
                    for subject in fold_data["outer_dev_subjects"]:
                        indices = bundle.indices([subject], (2,)); y = bundle.labels(indices)
                        le, ll = batched_logits(loaded["EEGNet"], indices, cache), batched_logits(loaded["LiteBN"], indices, cache)
                        pe, pl = le.argmax(axis=1), ll.argmax(axis=1)
                        prob_e = torch.softmax(torch.from_numpy(le), dim=1).numpy(); prob_l = torch.softmax(torch.from_numpy(ll), dim=1).numpy()
                        predictions = {"EEGNet":pe,"LiteBN":pl,"LOGIT50":(0.5*le + 0.5*ll).argmax(axis=1),"PROB50":(0.5*prob_e + 0.5*prob_l).argmax(axis=1)}
                        for name, prediction in predictions.items():
                            all_rows.append({"dataset":dataset,"fold":fold,"seed":seed,"checkpoint_variant":variant,"subject_id":str(subject),"method":name,"trials":int(len(y)),**metrics(y,prediction),"non_deployable":False})
                        e_correct, l_correct = pe == y, pl == y
                        oracle = e_correct | l_correct
                        c0 = float(oracle[y == 0].mean()); c1 = float(oracle[y == 1].mean())
                        comp_rows.append({"dataset":dataset,"fold":fold,"seed":seed,"checkpoint_variant":variant,"subject_id":str(subject),"trials":int(len(y)),"both_correct":int((e_correct & l_correct).sum()),"eegnet_only_correct":int((e_correct & ~l_correct).sum()),"litebn_only_correct":int((~e_correct & l_correct).sum()),"both_wrong":int((~e_correct & ~l_correct).sum()),"disagree":int((pe != pl).sum()),"oracle_class0":c0,"oracle_class1":c1})
                    del loaded
            del cache
            torch.cuda.empty_cache()
    terminal = make_outputs(pd.DataFrame(all_rows), pd.DataFrame(comp_rows), source_fold, outputs, protocols)
    print(terminal, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
