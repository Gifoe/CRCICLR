"""Repair-R2 source-only experiment: low-rank local target-conditional OT.

R2 reuses the audited V3/R1 loader, model, optimizer, and source split.  The
only scientific change is the local low-rank transport operator implemented in
``repair_r1._low_rank_local_displacement``.  Runtime artifacts remain on the
server; this module writes only compact source summaries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


BASE_REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
R1_CODE = BASE_REPO / "experiments" / "persist_eeg_scst_autonomous_repair" / "code"
if str(R1_CODE) not in sys.path:
    sys.path.insert(0, str(R1_CODE))

import repair_r1 as r1  # noqa: E402


EXP = BASE_REPO / "experiments" / "persist_eeg_scst_autonomous_repair"
CODE = EXP / "code"
PROTOCOL = EXP / "protocol"
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
RUNTIME = EXP / "runtime"

# Keep the shared common module pointed at R2 only for output; detached V2
# source caches and all data-access checks remain unchanged.
c = r1.c
c.EXP = EXP
c.CODE = CODE
c.PROTOCOL = PROTOCOL
c.RESULTS = RESULTS
c.FIGURES = FIGURES
c.RUNTIME = RUNTIME

PRIMARY = "R2-LowRank-Local-OT"
RANDOM = "R2-LowRank-Local-Random"
PREVIOUS = r1.PRIMARY
ERM = r1.ERM
MIXUP = r1.MIXUP
RECIPES = r1.RECIPES
ALPHA_LADDER = r1.ALPHA_LADDER
WARMUP_EPOCHS = int(c.WARMUP_EPOCHS)


def unit_dir(dataset: str, fold: int, seed: int) -> Path:
    return RUNTIME / "r2_units" / dataset / f"fold-{fold}" / f"seed-{seed}"


def run_unit(dataset: str, fold: int, seed: int, device: torch.device) -> None:
    directory = unit_dir(dataset, fold, seed)
    marker = directory / "COMPLETE.json"
    if marker.is_file():
        return
    train = c.load_feature_cache(dataset, fold, seed, "train")
    valid = c.load_feature_cache(dataset, fold, seed, "validation")
    directory.mkdir(parents=True, exist_ok=True)
    base_model, base_teacher = r1._warmup(train, dataset, fold, seed, device)
    with torch.inference_mode():
        base_features = base_teacher.features(torch.from_numpy(train["features"]).to(device)).float().cpu().numpy()
    bank = r1.BuresBank(base_features, train["labels"], train["subjects"], train["indices"], dataset=dataset, fold=fold, seed=seed)
    local, local_dirs = r1._geometry(
        base_features,
        train["labels"],
        train["subjects"],
        train["indices"],
        dataset,
        fold,
        seed,
        base_teacher,
        mode="local",
        bank=bank,
    )
    random_geometry, _ = r1._geometry(
        base_features,
        train["labels"],
        train["subjects"],
        train["indices"],
        dataset,
        fold,
        seed,
        base_teacher,
        mode="random",
        bank=bank,
        reference_directions=local_dirs,
    )
    (local_mask, random_mask), _ = r1._matched_masks(local, random_geometry, dataset, fold, seed)
    local.valid &= local_mask
    random_geometry.valid &= random_mask
    previous, _ = r1._geometry(
        base_features,
        train["labels"],
        train["subjects"],
        train["indices"],
        dataset,
        fold,
        seed,
        base_teacher,
        mode="protected",
        bank=bank,
    )

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    geometry_rows: list[pd.DataFrame] = []
    controls = [
        (ERM, 0.50, 0.50, None),
        (MIXUP, 0.50, 0.50, None),
        (PREVIOUS, 0.50, 0.50, previous),
    ]
    for method, q, lam, geometry in controls:
        frame, summary, _ = r1._train_from_warmup(base_model, base_teacher, train, valid, dataset, fold, seed, method, q, lam, geometry, device)
        frames.append(frame)
        summaries.append(summary)
    for q, lam in RECIPES:
        for method, geometry in ((PRIMARY, local), (RANDOM, random_geometry)):
            frame, summary, _ = r1._train_from_warmup(base_model, base_teacher, train, valid, dataset, fold, seed, method, q, lam, geometry, device)
            frames.append(frame)
            summaries.append(summary)
    geometry_rows.append(r1._summary_geometry(local, train, dataset, fold, seed, PRIMARY))
    geometry_rows.append(r1._summary_geometry(random_geometry, train, dataset, fold, seed, RANDOM))
    geometry_rows.append(r1._summary_geometry(previous, train, dataset, fold, seed, PREVIOUS))
    c.write_csv(directory / "per_subject.csv", pd.concat(frames, ignore_index=True))
    c.write_json(directory / "summary.json", summaries)
    c.write_csv(directory / "geometry_per_subject.csv", pd.concat(geometry_rows, ignore_index=True))
    c.write_json(directory / "match_audit.json", r1._match_audit(local, random_geometry, dataset, fold, seed))
    c.write_json(marker, {"dataset": dataset, "fold": fold, "seed": seed, "methods": len(summaries), "future_or_outer_opened": False})


def _paired(frame: pd.DataFrame, dataset: str, method: str, control: str, q: float, lam: float) -> np.ndarray:
    left = frame[(frame.dataset == dataset) & (frame.method == method) & np.isclose(frame.q, q) & np.isclose(frame.lambda_T, lam)]
    right = frame[(frame.dataset == dataset) & (frame.method == control)]
    if not len(left) or not len(right):
        return np.asarray([], np.float64)
    l = left.groupby("subject_id").BA.mean()
    r = right.groupby("subject_id").BA.mean()
    return (l - r).dropna().to_numpy(np.float64)


def _gate(frame: pd.DataFrame, geometry: pd.DataFrame, match: pd.DataFrame) -> dict[str, object]:
    recipe_rows: list[dict[str, object]] = []
    for q, lam in RECIPES:
        checks: dict[str, bool] = {}
        by_dataset: dict[str, dict[str, float | int | None]] = {}
        pooled_parts: list[np.ndarray] = []
        for dataset in c.DATASETS:
            delta = _paired(frame, dataset, PRIMARY, ERM, q, lam)
            ci = c.bootstrap_ci(delta, seed=c.stable_seed("r2-gate", dataset, q, lam)) if len(delta) else (float("nan"), float("nan"), float("nan"))
            random_delta = _paired(frame, dataset, PRIMARY, RANDOM, q, lam)
            random_ci = c.bootstrap_ci(random_delta, seed=c.stable_seed("r2-random-gate", dataset, q, lam)) if len(random_delta) else (float("nan"), float("nan"), float("nan"))
            mixup_delta = _paired(frame, dataset, PRIMARY, MIXUP, q, lam)
            pooled_parts.append(delta)
            by_dataset[dataset] = {
                "delta": ci[0],
                "ci95_l": ci[1],
                "ci95_u": ci[2],
                "n_subjects": int(len(delta)),
                "vs_random_delta": random_ci[0],
                "vs_random_ci95_l": random_ci[1],
                "vs_mixup_delta": float(mixup_delta.mean()) if len(mixup_delta) else None,
            }
            checks[f"{dataset}_delta_ge_002"] = bool(np.isfinite(ci[0]) and ci[0] >= 0.002)
            checks[f"{dataset}_ci_lower_vs_erm_positive"] = bool(np.isfinite(ci[1]) and ci[1] > 0)
            checks[f"{dataset}_ci_lower_vs_random_positive"] = bool(np.isfinite(random_ci[1]) and random_ci[1] > 0)
            checks[f"{dataset}_mean_vs_mixup_positive"] = bool(np.isfinite(by_dataset[dataset]["vs_mixup_delta"]) and float(by_dataset[dataset]["vs_mixup_delta"]) > 0)
        pooled = np.concatenate([part for part in pooled_parts if len(part)]) if any(len(part) for part in pooled_parts) else np.asarray([], np.float64)
        recipe_rows.append({
            "q": q,
            "lambda_T": lam,
            "by_dataset": by_dataset,
            "subject_nonnegative_fraction": float(np.mean(pooled >= 0)) if len(pooled) else 0.0,
            "checks": checks,
            "pass": bool(all(checks.values())),
        })

    affinity: dict[str, dict[str, object]] = {}
    primary_geometry = geometry[geometry.method == PRIMARY] if len(geometry) and "method" in geometry else geometry
    for dataset in c.DATASETS:
        subset = primary_geometry[primary_geometry.dataset == dataset]
        subject = subset.groupby("subject_id").agg(
            target_distance_improvement=("target_distance_improvement", "mean"),
            target_nll_improvement=("target_nll_improvement", "mean"),
            coverage=("coverage", "mean"),
            class_pass_rate=("class_pass_rate", "mean"),
            displacement_ratio=("median_displacement_ratio", "median"),
        ) if len(subset) else pd.DataFrame()
        if len(subject):
            dci = c.bootstrap_ci(subject.target_distance_improvement.to_numpy(float), seed=c.stable_seed("r2-affinity", dataset, "distance"))
            nci = c.bootstrap_ci(subject.target_nll_improvement.to_numpy(float), seed=c.stable_seed("r2-affinity", dataset, "nll"))
            affinity[dataset] = {
                "subjects": int(len(subject)),
                "target_distance_mean": dci[0],
                "target_distance_ci95_l": dci[1],
                "target_nll_mean": nci[0],
                "target_nll_ci95_l": nci[1],
                "coverage_mean": float(subject.coverage.mean()),
                "class_fidelity_mean": float(subject.class_pass_rate.mean()),
                "median_displacement_ratio": float(subject.displacement_ratio.median()),
                "checks": {
                    "target_affinity_ci_lower_positive": bool(dci[1] > 0 and nci[1] > 0),
                    "coverage": bool(subject.coverage.mean() >= 0.50),
                    "class_fidelity": bool(subject.class_pass_rate.mean() >= 0.90),
                    "displacement": bool(subject.displacement_ratio.median() >= 0.15),
                },
            }
        else:
            affinity[dataset] = {"subjects": 0, "checks": {"target_affinity_ci_lower_positive": False, "coverage": False, "class_fidelity": False, "displacement": False}}

    candidate_survival = float(primary_geometry.coverage.mean()) if len(primary_geometry) and "coverage" in primary_geometry else 0.0
    match_summary = {
        "rows": int(match.rows.sum()) if len(match) else 0,
        "matched_pairs": int(match.matched_pairs.sum()) if len(match) else 0,
        "mean_euclidean_norm_mismatch": float(np.nanmean(match.mean_euclidean_norm_mismatch)) if len(match) else None,
        "mean_whitened_norm_mismatch": float(np.nanmean(match.mean_whitened_norm_mismatch)) if len(match) else None,
        "alpha_mismatch_max": float(np.nanmax(match.alpha_mismatch_max)) if len(match) else None,
        "per_anchor_count_match": bool(match.per_anchor_count_match.all()) if len(match) else False,
    }
    source_pass = bool(
        any(row["pass"] and float(row["subject_nonnegative_fraction"]) >= 0.60 for row in recipe_rows)
        and all(
            value["checks"]["target_affinity_ci_lower_positive"]
            and value["checks"]["coverage"]
            and value["checks"]["class_fidelity"]
            and value["checks"]["displacement"]
            for value in affinity.values()
        )
        and candidate_survival <= 0.95
    )
    return {
        "schema": "SCST_AUTONOMOUS_R2_GATE_V1",
        "method": PRIMARY,
        "recipe_rows": recipe_rows,
        "affinity": affinity,
        "candidate_survival_mean": candidate_survival,
        "candidate_survival_guard": bool(candidate_survival <= 0.95),
        "match_audit": match_summary,
        "source_gate_pass": source_pass,
        "future_or_outer_opened": False,
        "outer_or_sealed_opened": False,
        "terminal_if_stop": "R2_SOURCE_GATE_PASSED" if source_pass else "R2_SOURCE_GATE_FAILED",
    }


def aggregate() -> None:
    files = sorted((RUNTIME / "r2_units").rglob("per_subject.csv"))
    if len(files) != len(c.DATASETS) * len(c.FOLDS) * len(c.SEEDS):
        raise RuntimeError(f"R2_INCOMPLETE_SOURCE_RESULTS:{len(files)}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    geometry = pd.concat([pd.read_csv(path) for path in sorted((RUNTIME / "r2_units").rglob("geometry_per_subject.csv"))], ignore_index=True)
    match = pd.DataFrame([json.loads(path.read_text(encoding="utf-8")) for path in sorted((RUNTIME / "r2_units").rglob("match_audit.json"))])
    c.write_csv(RESULTS / "R2_SOURCE_PER_SUBJECT.csv", frame)
    grouped = frame.groupby(["dataset", "method", "q", "lambda_T", "fold", "seed"], as_index=False).agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), subjects=("subject_id", "nunique"))
    c.write_csv(RESULTS / "R2_SOURCE_PER_FOLD.csv", grouped)
    c.write_csv(RESULTS / "R2_GEOMETRY_PER_SUBJECT.csv", geometry)
    c.write_csv(RESULTS / "R2_MATCH_AUDIT.csv", match)
    c.write_csv(RESULTS / "R2_METHOD_SUMMARY.csv", grouped)
    gate = _gate(frame, geometry, match)
    c.write_json(RESULTS / "R2_GATE.json", gate)
    stats = {
        "schema": "SCST_AUTONOMOUS_R2_STATISTICS_V1",
        "source_units": int(len(files)),
        "rows": int(len(frame)),
        "future_or_outer_opened": False,
        "outer_or_sealed_opened": False,
        "source_gate_pass": bool(gate["source_gate_pass"]),
        "terminal": gate["terminal_if_stop"],
    }
    c.write_json(RESULTS / "R2_STATISTICS.json", stats)
    print(json.dumps({"source_units": len(files), "source_gate_pass": gate["source_gate_pass"], "terminal": gate["terminal_if_stop"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=c.DATASETS)
    parser.add_argument("--fold", type=int, choices=c.FOLDS)
    parser.add_argument("--seed", type=int, choices=c.SEEDS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    c.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.all:
        for dataset in c.DATASETS:
            for fold in c.FOLDS:
                for seed in c.SEEDS:
                    print(f"[r2] START {dataset} f={fold} s={seed}", flush=True)
                    run_unit(dataset, fold, seed, device)
                    print(f"[r2] DONE {dataset} f={fold} s={seed}", flush=True)
    elif args.dataset is not None and args.fold is not None and args.seed is not None:
        run_unit(args.dataset, args.fold, args.seed, device)
    if args.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
