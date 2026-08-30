"""Run the pre-registered PERSIST-RE source-only recipe search and gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import persist_re_core as c


def recipes() -> list[dict[str, object]]:
    return [
        {"id": f"re_r{rank}_lr{lr:g}_lp{lp:g}", "rank": rank, "lambda_R": lr, "lambda_P": lp}
        for rank in c.SEARCH_RANKS for lr in c.SEARCH_LAMBDA_R for lp in c.SEARCH_LAMBDA_P
    ]


def subject_metric(frame: pd.DataFrame) -> float:
    return float(frame.groupby("subject_id", sort=False).BA.mean().mean())


def fit_cached(method: str, rep: dict[str, np.ndarray], dataset: str, fold: int, seed: int, recipe: dict[str, object], device: torch.device):
    # Include the training partition size so search fits (model-fit only) can
    # never be mistaken for the final source fit (model-fit + validation).
    key = f"{c.FIT_VERSION}_{dataset}_{method}_r{recipe['rank']}_lr{recipe['lambda_R']}_lp{recipe['lambda_P']}_f{fold}_s{seed}_n{len(rep['indices'])}"
    path = c.RUNTIME / "fits" / f"{key}.pt"
    subjects, _ = c.subject_index(rep["subjects"])
    if path.is_file():
        payload = torch.load(path, map_location=device, weights_only=False)
        model = c.PERSISTRE(rep["features"].shape[1], len(subjects), int(recipe["rank"])).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload.get("diagnostics", {})
    model, diagnostics = c.fit_model(method, rep, int(recipe["rank"]), float(recipe["lambda_R"]), float(recipe["lambda_P"]), c.stable_seed(dataset, method, fold, seed, recipe["id"]), device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "diagnostics": diagnostics}, path)
    return model, diagnostics


def search(dataset: str, device: torch.device) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for recipe in recipes():
        vals = []
        for fold in c.FOLDS:
            fd = c.load_fold(dataset, fold, 0)
            model, _ = fit_cached("PERSIST-RE", fd.model_fit, dataset, fold, 0, recipe, device)
            _, mapping = c.subject_index(fd.model_fit["subjects"])
            pred = c.predict(model, fd.validation, mapping, device)
            metric = c.metric_rows(dataset, fold, 0, "PERSIST-RE", pred)
            value = pd.DataFrame(metric)
            vals.append(subject_metric(value))
            rows.append({"dataset": dataset, "recipe": recipe["id"], "rank": recipe["rank"], "lambda_R": recipe["lambda_R"], "lambda_P": recipe["lambda_P"], "fold": fold, "seed": 0, "validation_BA": vals[-1], "validation_subjects": int(value.subject_id.nunique()), "future_session_used": False})
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["dataset", "recipe", "rank", "lambda_R", "lambda_P"], as_index=False).agg(mean_validation_BA=("validation_BA", "mean"), minimum_fold_BA=("validation_BA", "min"))
    selected_row = grouped.sort_values(["mean_validation_BA", "minimum_fold_BA", "recipe"], ascending=[False, False, True]).iloc[0]
    selected = next(r for r in recipes() if r["id"] == selected_row.recipe)
    c.write_csv(c.RESULTS / f"SOURCE_RECIPE_SEARCH_{dataset}.csv", frame)
    c.write_csv(c.RESULTS / f"SOURCE_RECIPE_SELECTION_{dataset}.csv", pd.DataFrame([{**selected, "dataset": dataset, **selected_row.to_dict(), "selected_before_outcome": True}]))
    return selected, rows


def bootstrap_delta(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    per_fold = frame.groupby(["dataset", "method", "fold", "seed"], as_index=False).agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), NLL=("NLL", "mean"), subjects=("subject_id", "nunique"))
    # Biological-subject bootstrap: first average repeated seeds/folds for each
    # subject, then resample subjects.  Trials are never bootstrap units.
    subject = frame.groupby(["dataset", "method", "subject_id"], as_index=False).agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), NLL=("NLL", "mean"))
    summaries = []
    deltas = []
    for dataset in c.DATASETS:
        ds = subject[subject.dataset == dataset]
        for method in c.METHODS:
            vals = ds[ds.method == method].set_index("subject_id").BA
            if vals.empty: continue
            mean, lo, hi = bootstrap_delta(vals.to_numpy(), c.stable_seed("source-bootstrap", dataset, method))
            summaries.append({"dataset": dataset, "method": method, "BA": mean, "CI95_L": lo, "CI95_U": hi, "subjects": int(vals.size)})
        erm = ds[ds.method == "ERM"].set_index("subject_id").BA
        for method in c.METHODS:
            if method == "ERM": continue
            other = ds[ds.method == method].set_index("subject_id").BA
            common = erm.index.intersection(other.index)
            if common.empty: continue
            delta, lo, hi = bootstrap_delta((other.loc[common] - erm.loc[common]).to_numpy(), c.stable_seed("source-paired-bootstrap", dataset, method))
            fold_delta = per_fold[(per_fold.dataset == dataset) & (per_fold.method == method)].set_index(["fold", "seed"]).BA - per_fold[(per_fold.dataset == dataset) & (per_fold.method == "ERM")].set_index(["fold", "seed"]).BA
            deltas.append({"dataset": dataset, "comparison": f"{method}-ERM", "delta_BA": delta, "CI95_L": lo, "CI95_U": hi, "positive_fold_seed_units": int((fold_delta > 0).sum()), "fold_seed_units": int(fold_delta.size)})
    return per_fold, subject, {"method_summary": summaries, "deltas": deltas}


def gate(statistics: dict[str, object], subject: pd.DataFrame) -> dict[str, object]:
    deltas = pd.DataFrame(statistics["deltas"])
    checks = {}
    for dataset in c.DATASETS:
        subset = deltas[deltas.dataset == dataset].set_index("comparison")
        full = subset.loc["PERSIST-RE-ERM"] if "PERSIST-RE-ERM" in subset.index else None
        checks[f"{dataset}_delta_ge_0_002"] = bool(full is not None and full.delta_BA >= 0.002)
        checks[f"{dataset}_paired_ci_lower_gt_zero"] = bool(full is not None and full.CI95_L > 0)
        for control in ("GroupDRO", "Mixup", "ProspectiveOnly", "RandomEffectOnly"):
            row = subset.loc[f"{control}-ERM"] if f"{control}-ERM" in subset.index else None
            checks[f"{dataset}_full_gt_{control}"] = bool(full is not None and row is not None and full.delta_BA > row.delta_BA)
    full_subject = subject[subject.method == "PERSIST-RE"].set_index("subject_id").BA
    erm_subject = subject[subject.method == "ERM"].set_index("subject_id").BA
    common = full_subject.index.intersection(erm_subject.index)
    fraction = float(np.mean((full_subject.loc[common] - erm_subject.loc[common]).to_numpy() >= 0)) if len(common) else 0.0
    checks["pooled_non_negative_subjects_ge_0_60"] = fraction >= 0.60
    checks["pooled_non_negative_subject_fraction"] = fraction
    diag_path = c.RESULTS / "RANDOM_EFFECT_STATISTICS.csv"
    diagnostics = pd.read_csv(diag_path) if diag_path.is_file() else pd.DataFrame()
    checks["random_effect_non_degenerate"] = bool(not diagnostics.empty and float(diagnostics.random_effect_parameter_norm.mean()) > 1e-6)
    checks["centered_effects_near_zero"] = bool(not diagnostics.empty and float(diagnostics.center_e_mean_norm.max()) < 1e-6 and float(diagnostics.center_a_mean_norm.max()) < 1e-6)
    checks["population_only_inference"] = True
    checks["future_session_used_during_source"] = False
    checks["source_gate_pass"] = bool(all(v is True for k, v in checks.items() if k.startswith(("OpenBMI_", "WBCIC_")) and k not in {"OpenBMI_pooled_non_negative_subject_fraction", "WBCIC_pooled_non_negative_subject_fraction"}) and checks["pooled_non_negative_subjects_ge_0_60"] and checks["random_effect_non_degenerate"] and checks["centered_effects_near_zero"])
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=c.DATASETS, default=list(c.DATASETS))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    c.RESULTS.mkdir(parents=True, exist_ok=True); c.RUNTIME.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    selections = {}
    search_rows = []
    for dataset in args.datasets:
        selected, rows = search(dataset, device)
        selections[dataset] = selected
        search_rows.extend(rows)
    c.write_csv(c.RESULTS / "SOURCE_RECIPE_SEARCH.csv", pd.DataFrame(search_rows))
    # Final source-only outcome table uses only the selected recipe and trains
    # on model-fit + validation.  Every matched method gets identical scope,
    # epochs, folds, and seeds.
    per_subject_rows = []
    diag_rows = []
    for dataset, recipe in selections.items():
        for fold in c.FOLDS:
            for seed in c.SEEDS:
                fd = c.load_fold(dataset, fold, seed)
                train = c.concat_rep(fd.model_fit, fd.validation)
                _, mapping = c.subject_index(train["subjects"])
                for method in c.METHODS:
                    model, diagnostics = fit_cached(method, train, dataset, fold, seed, recipe, device)
                    pred = c.predict(model, fd.outcome, mapping, device)
                    per_subject_rows.extend(c.metric_rows(dataset, fold, seed, method, pred))
                    diag_rows.append({"dataset": dataset, "fold": fold, "seed": seed, "method": method, "rank": recipe["rank"], "lambda_R": recipe["lambda_R"], "lambda_P": recipe["lambda_P"], "center_e_norm": diagnostics.get("center_e_norm", 0.0), "center_a_norm": diagnostics.get("center_a_norm", 0.0), "center_e_mean_norm": diagnostics.get("center_e_mean_norm", 0.0), "center_a_mean_norm": diagnostics.get("center_a_mean_norm", 0.0), "random_effect_variance": diagnostics.get("random_effect_variance", 0.0), "random_effect_parameter_norm": diagnostics.get("random_effect_parameter_norm", 0.0), "subject_count": diagnostics.get("subject_count", 0), "future_session_used": False})
                    del model
                    if device.type == "cuda": torch.cuda.empty_cache()
                print(f"[source-outcome] {dataset} fold={fold} seed={seed} recipe={recipe['id']}", flush=True)
    per_subject = pd.DataFrame(per_subject_rows)
    c.write_csv(c.RESULTS / "PER_SUBJECT.csv", per_subject)
    c.write_csv(c.RESULTS / "RANDOM_EFFECT_STATISTICS.csv", pd.DataFrame(diag_rows))
    per_fold, subject, statistics = summarize(per_subject)
    c.write_csv(c.RESULTS / "PER_FOLD.csv", per_fold)
    c.write_csv(c.RESULTS / "METHOD_SUMMARY.csv", pd.DataFrame(statistics["method_summary"]))
    c.write_csv(c.RESULTS / "ABLATION_SUMMARY.csv", pd.DataFrame(statistics["deltas"]))
    checks = gate(statistics, subject)
    c.write_json(c.RESULTS / "STATISTICS.json", statistics)
    c.write_json(c.RESULTS / "SOURCE_GATE.json", checks)
    print(json.dumps({"selections": selections, "gate": checks}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
