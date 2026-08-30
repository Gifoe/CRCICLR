"""Repair-R3 source-only experiment: task-protected local OT.

R3 keeps the frozen R2 local low-rank target-conditional operator and applies
one source-only rank-one task projection to its displacement.  All loaders,
models, controls, recipes, alpha ladder, and source gates are inherited from
the audited R1/R2 implementation.  Runtime artifacts stay on the server;
only compact summaries are written to the results directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


BASE_REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
EXP = BASE_REPO / "experiments" / "persist_eeg_scst_autonomous_repair"
CODE = EXP / "code"
PROTOCOL = EXP / "protocol"
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
RUNTIME = EXP / "runtime"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import repair_r1 as r1  # noqa: E402
import repair_r2 as r2  # noqa: E402


# Redirect the shared audited helpers to the R3 output namespace only.
c = r1.c
c.EXP = EXP
c.CODE = CODE
c.PROTOCOL = PROTOCOL
c.RESULTS = RESULTS
c.FIGURES = FIGURES
c.RUNTIME = RUNTIME
r2.c = c

PRIMARY = "R3-TaskProtected-Local-OT"
RANDOM = "R3-TaskProtected-Local-Random"
PREVIOUS = r1.PRIMARY
ERM = r1.ERM
MIXUP = r1.MIXUP
RECIPES = r2.RECIPES
ALPHA_LADDER = r2.ALPHA_LADDER
WARMUP_EPOCHS = int(c.WARMUP_EPOCHS)


def unit_dir(dataset: str, fold: int, seed: int) -> Path:
    return RUNTIME / "r3_units" / dataset / f"fold-{fold}" / f"seed-{seed}"


def _r3_geometry(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    row_ids: np.ndarray,
    dataset: str,
    fold: int,
    seed: int,
    teacher: r1.AdapterHead,
    bank: r1.BuresBank,
) -> tuple[r1.Geometry, np.ndarray]:
    """Build the R2 geometry after projecting its local displacement."""
    task_direction = r1._task_direction(features, labels)
    original = r1._low_rank_local_displacement

    def projected(bank_arg: r1.BuresBank, position: int, target: str) -> np.ndarray:
        local = original(bank_arg, position, target)
        return r1._protected(local, task_direction)

    # r1._geometry dispatches its local operator through the module global.
    # Patch only for this synchronous call and restore it unconditionally.
    r1._low_rank_local_displacement = projected
    try:
        return r1._geometry(
            features,
            labels,
            subjects,
            row_ids,
            dataset,
            fold,
            seed,
            teacher,
            mode="local",
            bank=bank,
        )
    finally:
        r1._low_rank_local_displacement = original


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
    local, local_dirs = _r3_geometry(
        base_features,
        train["labels"],
        train["subjects"],
        train["indices"],
        dataset,
        fold,
        seed,
        base_teacher,
        bank,
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


def _gate(frame: pd.DataFrame, geometry: pd.DataFrame, match: pd.DataFrame) -> dict[str, object]:
    # r2._gate is the frozen source-gate implementation.  Its thresholds and
    # same-recipe requirement are reused verbatim; only schema/method labels
    # are changed for an auditable R3 namespace.
    r2.PRIMARY = PRIMARY
    r2.RANDOM = RANDOM
    r2.PREVIOUS = PREVIOUS
    r2.ERM = ERM
    r2.MIXUP = MIXUP
    r2.RECIPES = RECIPES
    gate = r2._gate(frame, geometry, match)
    gate["schema"] = "SCST_AUTONOMOUS_R3_GATE_V1"
    gate["method"] = PRIMARY
    gate["terminal_if_stop"] = "R3_SOURCE_GATE_PASSED" if gate["source_gate_pass"] else "R3_SOURCE_GATE_FAILED"
    return gate


def aggregate() -> None:
    files = sorted((RUNTIME / "r3_units").rglob("per_subject.csv"))
    expected = len(c.DATASETS) * len(c.FOLDS) * len(c.SEEDS)
    if len(files) != expected:
        raise RuntimeError(f"R3_INCOMPLETE_SOURCE_RESULTS:{len(files)}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    geometry = pd.concat([pd.read_csv(path) for path in sorted((RUNTIME / "r3_units").rglob("geometry_per_subject.csv"))], ignore_index=True)
    match = pd.DataFrame([json.loads(path.read_text(encoding="utf-8")) for path in sorted((RUNTIME / "r3_units").rglob("match_audit.json"))])
    c.write_csv(RESULTS / "R3_SOURCE_PER_SUBJECT.csv", frame)
    grouped = frame.groupby(["dataset", "method", "q", "lambda_T", "fold", "seed"], as_index=False).agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), subjects=("subject_id", "nunique"))
    c.write_csv(RESULTS / "R3_SOURCE_PER_FOLD.csv", grouped)
    c.write_csv(RESULTS / "R3_GEOMETRY_PER_SUBJECT.csv", geometry)
    c.write_csv(RESULTS / "R3_MATCH_AUDIT.csv", match)
    c.write_csv(RESULTS / "R3_METHOD_SUMMARY.csv", grouped)
    gate = _gate(frame, geometry, match)
    c.write_json(RESULTS / "R3_GATE.json", gate)
    stats = {
        "schema": "SCST_AUTONOMOUS_R3_STATISTICS_V1",
        "source_units": int(len(files)),
        "rows": int(len(frame)),
        "future_or_outer_opened": False,
        "outer_or_sealed_opened": False,
        "source_gate_pass": bool(gate["source_gate_pass"]),
        "terminal": gate["terminal_if_stop"],
    }
    c.write_json(RESULTS / "R3_STATISTICS.json", stats)
    print(json.dumps({"source_units": len(files), "source_gate_pass": gate["source_gate_pass"], "terminal": gate["terminal_if_stop"]}, indent=2))


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
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
                    print(f"[r3] START {dataset} f={fold} s={seed}", flush=True)
                    run_unit(dataset, fold, seed, device)
                    print(f"[r3] DONE {dataset} f={fold} s={seed}", flush=True)
    elif args.dataset is not None and args.fold is not None and args.seed is not None:
        run_unit(args.dataset, args.fold, args.seed, device)
    if args.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
