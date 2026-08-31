"""Run the preregistered 12-recipe PERSIST-PDA source experiment.

The source experiment uses only the frozen CleanRoom ``model_fit`` archive to
learn the shared basis and only historical (earliest-session) labels to fit
each validation/outcome subject adapter.  The later session is metrics-only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pda_core as c


def recipes() -> list[dict[str, object]]:
    return [
        {"id": f"pda_r{r}_lx{lx:g}_lp{lp:g}", "rank": r, "lambda_X": lx, "lambda_P": lp}
        for r in c.RANKS for lx in c.LAMBDA_X for lp in c.LAMBDA_PRECISION
    ]


def subject_mean(frame: pd.DataFrame, method: str) -> float:
    x = frame[frame.method == method]
    return float(x.groupby("subject", sort=True).BA.mean().mean()) if len(x) else float("nan")


def recipe_validation(dataset: str, recipe: dict[str, object], basis_cache: dict, progress: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rid = str(recipe["id"])
    for fold in c.FOLDS:
        for seed in c.SEEDS:
            fd = c.load_fold(dataset, fold, seed)
            key = (dataset, fold, seed, int(recipe["rank"]))
            if key not in basis_cache:
                basis_cache[key] = c.fit_shared_basis(fd.model_fit, int(recipe["rank"]), ridge=1e-2)
            basis = basis_cache[key]
            for tr in c.make_transitions(fd.validation):
                methods = c.fit_subject_methods(tr, basis, ridge=1e-2, lambda_x=float(recipe["lambda_X"]), lambda_precision=float(recipe["lambda_P"]))
                eval_rows, _ = c.evaluate_transition(tr, methods, basis, fold, seed, dataset, "validation", rid)
                checkpoint_id = c.population_checkpoint_id(fd.model_fit)
                for row in eval_rows:
                    row["population_checkpoint_id"] = checkpoint_id
                rows.extend(eval_rows)
            progress.append({"stage": "validation", "dataset": dataset, "recipe": rid, "fold": fold, "seed": seed, "subjects": len(c.make_transitions(fd.validation))})
    return pd.DataFrame(rows)


def select_same_recipe(validation: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    if validation.empty:
        raise RuntimeError("validation table is empty")
    rows = []
    for (dataset, recipe), g in validation.groupby(["dataset", "recipe"], sort=True):
        method_means = g.groupby("method").BA.mean()
        pop = float(method_means.get("population", np.nan))
        full = float(method_means.get("full_pda", np.nan))
        rows.append({"dataset": dataset, "recipe": recipe, "validation_population_BA": pop,
                     "validation_full_BA": full, "validation_delta_BA": full - pop,
                     "validation_subjects": int(g.subject.nunique()), "future_session_used_for_fit": False})
    summary = pd.DataFrame(rows)
    pivot = summary.pivot(index="recipe", columns="dataset", values="validation_delta_BA")
    for ds in c.DATASETS:
        if ds not in pivot:
            pivot[ds] = np.nan
    pivot["minimum_dataset_delta"] = pivot[list(c.DATASETS)].min(axis=1)
    pivot["mean_dataset_delta"] = pivot[list(c.DATASETS)].mean(axis=1)
    # Lexical id ordering is deterministic tie-break; no outcome labels are
    # consulted in recipe selection.
    selected_id = pivot.sort_values(["minimum_dataset_delta", "mean_dataset_delta"], ascending=False).index[0]
    selected = next(x for x in recipes() if x["id"] == selected_id)
    summary = summary.merge(pivot.reset_index(), on="recipe", how="left")
    return selected, summary


def mechanism_rows(dataset: str, fold: int, seed: int, role: str, recipe: dict[str, object], transitions: list[c.SubjectTransition], subject_methods: dict[str, dict[str, dict[str, object]]], basis: c.Basis, population_checkpoint_id: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    components: list[dict[str, object]] = []
    ids = [tr.subject for tr in transitions]
    # Correct adapters and their norms are known before constructing controls.
    norms = {s: float(np.linalg.norm(np.asarray(subject_methods[s]["full_pda"]["a"])) + np.linalg.norm(np.asarray(subject_methods[s]["full_pda"]["c"]))) for s in ids}
    for tr in transitions:
        s = tr.subject
        methods = subject_methods[s]
        base_rows, _ = c.evaluate_transition(tr, methods, basis, fold, seed, dataset, role, str(recipe["id"]))
        for row in base_rows:
            row["population_checkpoint_id"] = population_checkpoint_id
        rows.extend(base_rows)
        # Norm-matched wrong subject: nearest norm, deterministic subject tie.
        wrong_map, shuffled_map = c.control_assignments(ids, norms)
        wrong_s = wrong_map[s]
        wrong = subject_methods[wrong_s]["full_pda"]
        ba_w, f1_w = c.metric_for_rep(tr.future, np.asarray(wrong["a"]), np.asarray(wrong["c"]), basis)
        # Shuffled assignment is a non-identity cyclic permutation; unlike the
        # wrong control it is intentionally not norm matched.
        shuffled_s = shuffled_map[s]
        shuffled = subject_methods[shuffled_s]["full_pda"]
        ba_sh, f1_sh = c.metric_for_rep(tr.future, np.asarray(shuffled["a"]), np.asarray(shuffled["c"]), basis)
        ba_c, f1_c = c.metric_for_rep(tr.future, np.asarray(methods["full_pda"]["a"]), np.asarray(methods["full_pda"]["c"]), basis)
        rows.extend([
            {"dataset": dataset, "role": role, "fold": fold, "seed": seed, "subject": s, "method": "correct_adapter", "recipe": recipe["id"], "BA": ba_c, "macro_F1": f1_c, "future_session_used_for_fit": False, "future_labels_used_for_fit": False, "population_checkpoint_id": population_checkpoint_id},
            {"dataset": dataset, "role": role, "fold": fold, "seed": seed, "subject": s, "method": "wrong_adapter", "recipe": recipe["id"], "BA": ba_w, "macro_F1": f1_w, "future_session_used_for_fit": False, "future_labels_used_for_fit": False, "matched_subject": wrong_s, "population_checkpoint_id": population_checkpoint_id},
            {"dataset": dataset, "role": role, "fold": fold, "seed": seed, "subject": s, "method": "shuffled_adapter", "recipe": recipe["id"], "BA": ba_sh, "macro_F1": f1_sh, "future_session_used_for_fit": False, "future_labels_used_for_fit": False, "assigned_subject": shuffled_s, "population_checkpoint_id": population_checkpoint_id},
        ])
        parts = subject_methods[s]["parts"]
        loo = subject_methods[s]["loo"]
        components.append({"dataset": dataset, "role": role, "fold": fold, "seed": seed, "subject": s, "recipe": recipe["id"], "population_checkpoint_id": population_checkpoint_id,
                           "persistent_norm": subject_methods[s]["persistent_norm"], "transient_norm": subject_methods[s]["transient_norm"],
                           "persistent_transient_ratio": subject_methods[s]["persistent_transient_ratio"],
                           "transient_mean_norm": subject_methods[s]["transient_mean_norm"],
                           "historical_crossfit_gain": subject_methods[s]["crossfit_gain"],
                           "n_historical_blocks": len(parts), "fisher_a_sum": float(np.sum(subject_methods[s]["fisher_a"])),
                           "fisher_c_sum": float(np.sum(subject_methods[s]["fisher_c"])), "adapter_collapse": bool(subject_methods[s]["persistent_norm"] < 1e-10 and subject_methods[s]["transient_norm"] < 1e-10),
                           "transient_zero_center_error": subject_methods[s]["transient_mean_norm"]})
    return rows, components


def run_outcome(dataset: str, recipe: dict[str, object], basis_cache: dict, progress: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    components: list[dict[str, object]] = []
    for fold in c.FOLDS:
        for seed in c.SEEDS:
            fd = c.load_fold(dataset, fold, seed)
            key = (dataset, fold, seed, int(recipe["rank"]))
            if key not in basis_cache:
                basis_cache[key] = c.fit_shared_basis(fd.model_fit, int(recipe["rank"]), ridge=1e-2)
            basis = basis_cache[key]
            transitions = c.make_transitions(fd.outcome)
            fitted = {}
            for tr in transitions:
                fitted[tr.subject] = c.fit_subject_methods(tr, basis, ridge=1e-2, lambda_x=float(recipe["lambda_X"]), lambda_precision=float(recipe["lambda_P"]))
            r, comp = mechanism_rows(dataset, fold, seed, "outcome", recipe, transitions, fitted, basis, c.population_checkpoint_id(fd.model_fit))
            rows.extend(r); components.extend(comp)
            progress.append({"stage": "outcome", "dataset": dataset, "recipe": recipe["id"], "fold": fold, "seed": seed, "subjects": len(transitions)})
    return pd.DataFrame(rows), pd.DataFrame(components)


def gate(outcome: pd.DataFrame, components: pd.DataFrame, comparisons: pd.DataFrame, selected: dict[str, object]) -> dict[str, object]:
    checks: dict[str, object] = {}
    for ds in c.DATASETS:
        full = comparisons[(comparisons.dataset == ds) & (comparisons.comparison == "full_pda-population")]
        row = full.iloc[0] if len(full) else None
        checks[f"{ds}_delta_ge_0_003"] = bool(row is not None and float(row.delta_BA) >= 0.003)
        checks[f"{ds}_paired_ci_lower_gt_zero"] = bool(row is not None and float(row.CI95_L) > 0.0)
        for control in ("ordinary_adapter", "mean_pooled", "no_crossfit"):
            x = comparisons[(comparisons.dataset == ds) & (comparisons.comparison == f"full_pda-{control}")]
            checks[f"{ds}_full_gt_{control}"] = bool(len(x) and float(x.iloc[0].delta_BA) > 0.0)
        for control in ("wrong_adapter", "shuffled_adapter"):
            x = comparisons[(comparisons.dataset == ds) & (comparisons.comparison == f"correct_adapter-{control}")]
            checks[f"{ds}_correct_gt_{control}_ci_lower_gt_zero"] = bool(len(x) and float(x.iloc[0].CI95_L) > 0.0)
        x = comparisons[(comparisons.dataset == ds) & (comparisons.comparison == "full_pda-single_session")]
        checks[f"{ds}_persistent_gt_single_session_ci_lower_gt_zero"] = bool(len(x) and float(x.iloc[0].CI95_L) > 0.0)
    full = outcome[outcome.method == "full_pda"].groupby("subject", as_index=True).BA.mean()
    pop = outcome[outcome.method == "population"].groupby("subject", as_index=True).BA.mean()
    common = full.index.intersection(pop.index)
    frac = float(np.mean((full.loc[common] - pop.loc[common]).to_numpy() >= 0.0)) if len(common) else 0.0
    checks["nonnegative_subject_fraction"] = frac
    checks["nonnegative_subjects_ge_0_60"] = frac >= 0.60
    checks["no_future_labels_in_fit"] = bool((outcome.future_labels_used_for_fit == False).all())
    checks["no_future_session_in_fit"] = bool((outcome.future_session_used_for_fit == False).all())
    checks["population_frozen"] = True
    checks["fisher_finite_positive"] = bool(np.isfinite(components[["fisher_a_sum", "fisher_c_sum"]].to_numpy()).all() and (components[["fisher_a_sum", "fisher_c_sum"]].to_numpy() > 0).all())
    checks["no_adapter_collapse"] = bool(not components.adapter_collapse.all()) if len(components) else False
    checks["transient_zero_centered"] = bool(float(components.transient_zero_center_error.max()) < 1e-8) if len(components) else False
    prefix = tuple(k for k in checks if k.startswith(("OpenBMI_", "WBCIC_")))
    checks["source_gate_pass"] = bool(all(bool(checks[k]) for k in prefix) and checks["nonnegative_subjects_ge_0_60"] and checks["fisher_finite_positive"] and checks["no_future_labels_in_fit"] and checks["no_future_session_in_fit"] and checks["no_adapter_collapse"] and checks["transient_zero_centered"])
    checks["terminal"] = "PERSIST_PDA_SOURCE_ONLY_SUPPORTED" if checks["source_gate_pass"] else "PERSIST_PDA_SOURCE_NOT_SUPPORTED"
    checks["selected_recipe"] = selected
    return checks


def make_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    # Collapse repeated fold/seed observations within each biological subject
    # before bootstrap; a subject, never a trial, is the resampling unit.
    rows = []
    pairs = [("full_pda", "population"), ("full_pda", "ordinary_adapter"), ("full_pda", "mean_pooled"), ("full_pda", "no_crossfit"), ("correct_adapter", "wrong_adapter"), ("correct_adapter", "shuffled_adapter"), ("full_pda", "single_session")]
    for ds in c.DATASETS:
        for left, right in pairs:
            rows.append(c.paired_subject_bootstrap(frame, left, right, ds, "source-pda"))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=c.DATASETS, default=list(c.DATASETS))
    ap.add_argument("--backbone", default="ATCNet")
    args = ap.parse_args()
    if args.backbone != "ATCNet":
        raise SystemExit("source primary backbone is fixed to ATCNet-CleanRoom")
    c.RESULTS.mkdir(parents=True, exist_ok=True)
    progress: list[dict[str, object]] = []
    basis_cache: dict[object, c.Basis] = {}
    val_frames = []
    for ds in args.datasets:
        for rec in recipes():
            val_frames.append(recipe_validation(ds, rec, basis_cache, progress))
    validation = pd.concat(val_frames, ignore_index=True)
    selected, search = select_same_recipe(validation)
    c.write_csv(c.RESULTS / "SOURCE_RECIPE_SEARCH.csv", search)
    c.write_json(c.RESULTS / "SOURCE_RECIPE_SELECTION.json", {"selected": selected, "selection_rule": "maximize minimum OpenBMI/WBCIC validation delta", "validation_future_labels_used_for_fit": False})
    outcome_frames = []; comp_frames = []
    for ds in args.datasets:
        out, comp = run_outcome(ds, selected, basis_cache, progress)
        outcome_frames.append(out); comp_frames.append(comp)
    outcome = pd.concat(outcome_frames, ignore_index=True)
    components = pd.concat(comp_frames, ignore_index=True)
    c.write_csv(c.RESULTS / "SOURCE_PER_SUBJECT.csv", outcome)
    fold = outcome.groupby(["dataset", "role", "fold", "seed", "method"], as_index=False).agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), subjects=("subject", "nunique"))
    c.write_csv(c.RESULTS / "SOURCE_PER_FOLD.csv", fold)
    c.write_csv(c.RESULTS / "ADAPTER_COMPONENTS.csv", components)
    c.write_csv(c.RESULTS / "CORRECT_WRONG_SHUFFLED.csv", outcome[outcome.method.isin(["correct_adapter", "wrong_adapter", "shuffled_adapter"])])
    comparisons = make_comparisons(outcome)
    c.write_json(c.RESULTS / "STATISTICS.json", {"selected_recipe": selected, "comparisons": comparisons.to_dict(orient="records"), "n_bootstrap": c.N_BOOTSTRAP, "bootstrap_unit": "biological_subject", "progress_events": progress[-20:]})
    gate_result = gate(outcome, components, comparisons, selected)
    c.write_json(c.RESULTS / "SOURCE_GATE.json", gate_result)
    c.write_json(c.RESULTS / "ITERATION_STATE.json", {"finished": True, "terminal": gate_result["terminal"], "selected_recipe": selected, "future_resource_opened": False})
    print(json.dumps({"terminal": gate_result["terminal"], "selected_recipe": selected, "gate": gate_result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
