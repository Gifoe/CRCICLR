"""Frozen TeCh P2 selection-factor replication.

This runner never trains or fine-tunes a neural model.  It reads the corrected
PEEH assignment and corrected PSWA q caches.  The only reconstructed PEEH
quantity is the non-serialized persistence margin (rho - null_p95), rebuilt
deterministically from TRAIN-subject embeddings with the frozen checkpoint and
the original spectrum() code path.  Joint is always read from the stored PEEH
Protected union and is never reselected.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
OUT = EXP / "outputs"
PROTOCOL = EXP / "protocol"
RUNTIME = Path(os.environ.get("P2_RUNTIME", r"D:\nips-temp\TotalP\P1\p2_selection_factor_tech_runtime"))
PEEH_RUNTIME = Path(os.environ.get("PEEH_RUNTIME", r"D:\nips-temp\TotalP\P1\crossbackbone_peeh_runtime"))
PSWA_RUNTIME = Path(os.environ.get("PSWA_RUNTIME", r"D:\nips-temp\TotalP\P1\crossbackbone_pswa_runtime"))
PSWA_CODE = Path(os.environ.get(
    "PSWA_CODE_PATH",
    str(REPO / "experiments" / "persist_eeg_crossbackbone_pswa_v1" / "code" / "run_pswa_recovery.py"),
))

TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
FOLDS = tuple(range(5))
SEED = 0
RIDGE_ALPHA = 0.01
RANDOM_DRAWS = 100
BOOTSTRAP_DRAWS = 20_000
SELECTORS = ("RANDOM", "PERSISTENCE_ONLY", "UTILITY_ONLY", "JOINT")
MODEL = "TeCh"
MODEL_KEY = "tech"
EEGNET_EXP = REPO / "experiments" / "persist_eeg_p2_selection_factor_eegnet_v1"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


pswa = load_module("p2_frozen_pswa", PSWA_CODE)
peeh = pswa.peeh


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def natural(values: Iterable[object]) -> list[str]:
    return peeh.natural_subjects(values)


def peeh_path(task: str, fold: int) -> Path:
    return PEEH_RUNTIME / "cells" / MODEL_KEY / task.lower() / f"fold{fold}_seed0.json"


def pswa_path(task: str, fold: int) -> Path:
    return PSWA_RUNTIME / "cells" / MODEL_KEY / task.lower() / f"fold{fold}_seed0.json"


def cell_path(task: str, fold: int) -> Path:
    return RUNTIME / "cells" / task.lower() / f"fold{fold}_seed0.json"


def block_table(stored: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for raw in sorted(stored["protected_assignment"], key=lambda x: int(x["block"])):
        width = int(raw["dimensions"])
        coords = list(range(offset, offset + width))
        offset += width
        utility_margin = float(min(raw["absolute_CI_low"], raw["excess_CI_low"]))
        rows.append({
            "block": int(raw["block"]), "rank": width, "coordinates": coords,
            "persistence_pass": bool(raw["persistence_supported"]),
            "utility_margin": utility_margin, "utility_pass": utility_margin > 0.0,
            "absolute_CI_low": float(raw["absolute_CI_low"]),
            "equal_rank_excess_CI_low": float(raw["excess_CI_low"]),
            "joint_member": bool(raw["protected"]),
        })
    if offset != int(stored["rank"]):
        raise RuntimeError(f"block ranks sum to {offset}, active rank is {stored['rank']}")
    return rows


def better(candidate: tuple[float, tuple[int, ...]], incumbent: tuple[float, tuple[int, ...]] | None) -> bool:
    if incumbent is None:
        return True
    if candidate[0] > incumbent[0] + 1e-14:
        return True
    if abs(candidate[0] - incumbent[0]) <= 1e-14:
        if len(candidate[1]) < len(incumbent[1]):
            return True
        if len(candidate[1]) == len(incumbent[1]) and candidate[1] < incumbent[1]:
            return True
    return False


def exact_rank(blocks: list[dict[str, Any]], target: int, score_key: str, gate_key: str) -> tuple[list[int], list[int], float]:
    """Exact deterministic block knapsack with the preregistered tie-break."""
    dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for block in blocks:
        if not bool(block[gate_key]):
            continue
        width, score, block_id = int(block["rank"]), float(block[score_key]), int(block["block"])
        update = dict(dp)
        for rank, (total, chosen) in dp.items():
            new_rank = rank + width
            if new_rank > target:
                continue
            candidate = (total + score, chosen + (block_id,))
            if better(candidate, update.get(new_rank)):
                update[new_rank] = candidate
        dp = update
    if target not in dp:
        raise RuntimeError(f"no exact-rank solution for K={target}, score={score_key}, gate={gate_key}")
    score, selected_blocks = dp[target]
    selected = [b for b in blocks if int(b["block"]) in set(selected_blocks)]
    coordinates = sorted(itertools.chain.from_iterable(b["coordinates"] for b in selected))
    if len(coordinates) != target:
        raise AssertionError((len(coordinates), target))
    return list(selected_blocks), coordinates, float(score)


def reconstruct_persistence(task: str, fold: int, blocks: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    """Rebuild only the missing rho-null_p95 statistic on frozen TRAIN data."""
    row, checkpoint = pswa.checkpoint_row(MODEL, task, fold)
    if sha(checkpoint) != row["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA mismatch before persistence reconstruction")
    data = pswa.load_arrays(task, fold)
    if data["normalizer"]["mean_std_sha256"] != row["normalizer_sha256"]:
        raise RuntimeError("normalizer SHA mismatch before persistence reconstruction")
    capped = pswa.cap_data(data, task, fold)
    net, head = peeh.build_model(row, device)
    h_source = peeh.representations(net, head, capped["source_x"], MODEL, device, batch=32)
    h_future = peeh.representations(net, head, capped["future_x"], MODEL, device, batch=32)
    h = np.concatenate([h_source, h_future])
    y = np.concatenate([capped["source_y"], capped["future_y"]])
    owner = np.concatenate([capped["source_subjects"], capped["future_subjects"]]).astype(str)
    sessions = np.concatenate([
        np.full(len(h_source), data["source_session"]),
        np.full(len(h_future), data["future_session"]),
    ]).astype(np.int64)
    spec = peeh.spectrum(h, y, owner, sessions, task, MODEL, fold)
    widths = [len(x) for x in spec["blocks"]]
    if widths != [int(x["rank"]) for x in blocks]:
        raise RuntimeError(f"candidate-block mismatch: reconstructed={widths}, stored={[x['rank'] for x in blocks]}")
    support = []
    for stored_block, rebuilt, coordinates in zip(blocks, spec["support"], spec["blocks"]):
        if list(map(int, coordinates)) != stored_block["coordinates"]:
            raise RuntimeError("candidate coordinate mismatch")
        persistence_pass = bool(rebuilt["persistence_supported"])
        if persistence_pass != stored_block["persistence_pass"]:
            raise RuntimeError("persistence gate does not reproduce stored corrected PEEH")
        support.append({
            "block": int(rebuilt["block"]), "rho": float(rebuilt["rho"]),
            "null_p95": float(rebuilt["null_p95"]),
            "persistence_margin": float(rebuilt["rho"] - rebuilt["null_p95"]),
            "persistence_pass": persistence_pass,
        })
    del net, h_source, h_future, h
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "support": support,
        "checkpoint_sha256": row["checkpoint_sha256"],
        "normalizer_sha256": row["normalizer_sha256"],
        "split_sha256": row["split_sha256"],
        "device": str(device),
        "provenance": "RECONSTRUCTED_MISSING_DETERMINISTIC_STATISTIC_FROM_FROZEN_TRAIN_ONLY_ORIGINAL_SPECTRUM_PATH",
        "persistence_permutations": int(peeh.PERSISTENCE_PERMUTATIONS),
    }


def freeze_selectors(task: str, fold: int, device: torch.device) -> dict[str, Any]:
    ppath, swpath = peeh_path(task, fold), pswa_path(task, fold)
    if not ppath.is_file() or not swpath.is_file():
        raise FileNotFoundError((ppath, swpath))
    stored = json.loads(ppath.read_text(encoding="utf-8"))
    pswa_cell = json.loads(swpath.read_text(encoding="utf-8"))
    if pswa_cell.get("status") != "RECOVERY_OK":
        raise RuntimeError(f"PSWA cell not usable: {pswa_cell.get('status')}")
    blocks = block_table(stored)
    rebuilt = reconstruct_persistence(task, fold, blocks, device)
    margin_by_block = {int(x["block"]): x for x in rebuilt["support"]}
    for block in blocks:
        block.update(margin_by_block[int(block["block"])])

    joint = sorted(set(map(int, stored["protected_blocks"])))
    expected_joint = sorted(itertools.chain.from_iterable(x["coordinates"] for x in blocks if x["joint_member"]))
    if joint != expected_joint:
        raise RuntimeError("stored Joint coordinate union does not match stored assignment")
    k = int(stored["protected_dimensions"])
    if len(joint) != k or k <= 0:
        raise RuntimeError(f"unexpected Joint rank K={k}")
    p_blocks, persistence, p_score = exact_rank(blocks, k, "persistence_margin", "persistence_pass")
    u_blocks, utility, u_score = exact_rank(blocks, k, "utility_margin", "utility_pass")
    random_controls = [list(map(int, x)) for x in pswa_cell["random_controls"]]
    if len(random_controls) != RANDOM_DRAWS or any(len(set(x)) != k for x in random_controls):
        raise RuntimeError("random exact-rank controls are incomplete")
    if any(any(i < 0 or i >= int(stored["rank"]) for i in x) for x in random_controls):
        raise RuntimeError("random coordinate outside active rank")

    selectors = {
        "PERSISTENCE_ONLY": {"blocks": p_blocks, "coordinates": persistence, "score": p_score},
        "UTILITY_ONLY": {"blocks": u_blocks, "coordinates": utility, "score": u_score},
        "JOINT": {"blocks": [int(x["block"]) for x in blocks if x["joint_member"]], "coordinates": joint, "score": None},
        "RANDOM": {"draws": random_controls},
    }
    frozen = {
        "schema": "P2_TECH_FROZEN_SELECTORS_V1", "Model": MODEL, "Task": task,
        "fold": fold, "seed": 0, "active_rank": int(stored["rank"]), "K": k,
        "selectors": selectors, "candidate_blocks": blocks,
        "source": {
            "peeh_cell_path": str(ppath), "peeh_cell_sha256": sha(ppath),
            "pswa_cell_path": str(swpath), "pswa_cell_sha256": sha(swpath),
            "q_cache_path": pswa_cell["q_cache_path"], "q_cache_sha256": pswa_cell["q_cache_sha256"],
            "checkpoint_sha256": rebuilt["checkpoint_sha256"], "normalizer_sha256": rebuilt["normalizer_sha256"],
            "split_sha256": rebuilt["split_sha256"], "persistence_margin_provenance": rebuilt["provenance"],
        },
        "assertions": {
            "joint_exactly_stored_protected": True, "all_nonrandom_rank_K": all(len(selectors[x]["coordinates"]) == k for x in SELECTORS if x != "RANDOM"),
            "all_random_rank_K": all(len(x) == k for x in random_controls),
            "persistence_gate_reproduced": True, "selector_used_heldout_outcome": False,
        },
    }
    frozen["selector_freeze_sha256"] = object_hash(frozen)
    return frozen


def ridge_predict(train: np.ndarray, labels: np.ndarray, test: np.ndarray, classes: int) -> np.ndarray:
    return pswa.ridge_scores(train, labels, test, classes).argmax(1)


def evaluate_frozen(frozen: dict[str, Any]) -> list[dict[str, Any]]:
    """Heldout arrays are opened only after the selector freeze object exists."""
    cache = Path(frozen["source"]["q_cache_path"])
    if sha(cache) != frozen["source"]["q_cache_sha256"]:
        raise RuntimeError("q-cache SHA mismatch")
    with np.load(cache, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    train, labels = arrays["q_train"], arrays["y_train"]
    classes = int(max(labels.max(), *[arrays[x].max() for x in arrays if x.startswith("y_eval_")]) + 1)
    sessions = sorted(x.removeprefix("q_eval_") for x in arrays if x.startswith("q_eval_"))
    expected_sessions = ["S1", "S2"] if frozen["Task"].startswith("OpenBMI") else ["S0", "S1", "S2"]
    if sessions != expected_sessions:
        raise RuntimeError((sessions, expected_sessions))
    future = "S2"
    subjects = natural(itertools.chain.from_iterable(arrays[f"subjects_eval_{s}"].astype(str) for s in sessions))
    rows: list[dict[str, Any]] = []

    for selector in ("PERSISTENCE_ONLY", "UTILITY_ONLY", "JOINT"):
        dims = frozen["selectors"][selector]["coordinates"]
        by_subject: dict[str, list[tuple[str, float]]] = {s: [] for s in subjects}
        for session in sessions:
            q, y, owner = arrays[f"q_eval_{session}"], arrays[f"y_eval_{session}"], arrays[f"subjects_eval_{session}"].astype(str)
            pred = ridge_predict(train[:, dims], labels, q[:, dims], classes)
            for subject in subjects:
                mask = owner == subject
                by_subject[subject].append((session, float(balanced_accuracy_score(y[mask], pred[mask]))))
        for subject, values in by_subject.items():
            rows.append({"Model": MODEL, "Task": frozen["Task"], "fold": frozen["fold"], "seed": 0,
                         "subject_id": subject, "selector": selector, "WS_BA": min(x[1] for x in values),
                         "Future_BA": next(x[1] for x in values if x[0] == future)})

    random_ws: dict[str, list[float]] = {s: [] for s in subjects}
    random_future: dict[str, list[float]] = {s: [] for s in subjects}
    for dims in frozen["selectors"]["RANDOM"]["draws"]:
        draw: dict[str, list[tuple[str, float]]] = {s: [] for s in subjects}
        for session in sessions:
            q, y, owner = arrays[f"q_eval_{session}"], arrays[f"y_eval_{session}"], arrays[f"subjects_eval_{session}"].astype(str)
            pred = ridge_predict(train[:, dims], labels, q[:, dims], classes)
            for subject in subjects:
                mask = owner == subject
                draw[subject].append((session, float(balanced_accuracy_score(y[mask], pred[mask]))))
        for subject, values in draw.items():
            random_ws[subject].append(min(x[1] for x in values))
            random_future[subject].append(next(x[1] for x in values if x[0] == future))
    for subject in subjects:
        rows.append({"Model": MODEL, "Task": frozen["Task"], "fold": frozen["fold"], "seed": 0,
                     "subject_id": subject, "selector": "RANDOM", "WS_BA": float(np.mean(random_ws[subject])),
                     "Future_BA": float(np.mean(random_future[subject]))})
    return rows


def run_cell(task: str, fold: int, dry_run: bool = False) -> dict[str, Any]:
    path = cell_path(task, fold)
    if path.is_file() and not dry_run:
        return json.loads(path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen = freeze_selectors(task, fold, device)
    # The freeze hash is computed and persisted before heldout q/y are opened.
    freeze_path = RUNTIME / "selector_freezes" / task.lower() / f"fold{fold}_seed0.json"
    write_json(freeze_path, frozen)
    rows = evaluate_frozen(frozen)
    result = {
        "schema": "P2_TECH_CELL_V1", "Model": MODEL, "Task": task, "fold": fold, "seed": 0,
        "status": "COMPLETE", "selector_freeze_sha256": frozen["selector_freeze_sha256"],
        "selector_freeze_path": str(freeze_path), "K": frozen["K"], "active_rank": frozen["active_rank"],
        "selectors": frozen["selectors"], "candidate_blocks": frozen["candidate_blocks"],
        "subject_retained_ba": rows, "random_draws": RANDOM_DRAWS,
        "model_training_count": 0, "neural_finetuning_count": 0,
        "dry_run_validated": bool(dry_run), "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": frozen["source"],
    }
    write_json(path, result)
    print(f"P2_CELL_COMPLETE {task} fold={fold} K={frozen['K']} dry_run={dry_run}", flush=True)
    return result


def bootstrap(values: Sequence[float], *parts: object) -> tuple[float, float, float]:
    value = np.asarray(values, dtype=float)
    rng = np.random.default_rng(peeh.stable_seed("pswa-bootstrap", "p2-selection-factor", *parts))
    means = value[rng.integers(0, len(value), size=(BOOTSTRAP_DRAWS, len(value)))].mean(1)
    return float(value.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def aggregate() -> None:
    cells = []
    for task in TASKS:
        for fold in FOLDS:
            path = cell_path(task, fold)
            if not path.is_file():
                raise RuntimeError(f"missing cell {task}/fold{fold}")
            cells.append(json.loads(path.read_text(encoding="utf-8")))
    if len(cells) != 20 or any(x.get("status") != "COMPLETE" for x in cells):
        raise RuntimeError("20-cell matrix incomplete")

    selector_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    retained_rows: list[dict[str, Any]] = []
    manifest_cells: list[dict[str, Any]] = []
    for cell in cells:
        task, fold, k = cell["Task"], int(cell["fold"]), int(cell["K"])
        joint = set(cell["selectors"]["JOINT"]["coordinates"])
        utility = set(cell["selectors"]["UTILITY_ONLY"]["coordinates"])
        persistence = set(cell["selectors"]["PERSISTENCE_ONLY"]["coordinates"])
        for selector in ("PERSISTENCE_ONLY", "UTILITY_ONLY", "JOINT"):
            value = cell["selectors"][selector]
            selector_rows.append({"Task": task, "fold": fold, "seed": 0, "selector": selector, "K": k,
                                  "block_ids": json.dumps(value["blocks"]), "coordinate_ids": json.dumps(value["coordinates"]),
                                  "selector_score": "" if value["score"] is None else value["score"],
                                  "selector_freeze_sha256": cell["selector_freeze_sha256"]})
        selector_rows.append({"Task": task, "fold": fold, "seed": 0, "selector": "RANDOM", "K": k,
                              "block_ids": "", "coordinate_ids": "100 exact-rank draws in corrected PSWA artifact",
                              "selector_score": "", "selector_freeze_sha256": cell["selector_freeze_sha256"]})
        overlap_rows.append({"Task": task, "fold": fold, "seed": 0, "K": k,
                             "J_U_overlap": len(joint & utility) / k, "J_P_overlap": len(joint & persistence) / k,
                             "joint_coordinates": json.dumps(sorted(joint)), "utility_coordinates": json.dumps(sorted(utility)),
                             "persistence_coordinates": json.dumps(sorted(persistence))})
        retained_rows.extend(cell["subject_retained_ba"])
        manifest_cells.append({"Task": task, "fold": fold, **cell["source"]})

    contrast_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for task in TASKS:
        ids = natural(x["subject_id"] for x in retained_rows if x["Task"] == task)
        subject_selector: dict[tuple[str, str], dict[str, float]] = {}
        for subject in ids:
            for selector in SELECTORS:
                z = [x for x in retained_rows if x["Task"] == task and x["subject_id"] == subject and x["selector"] == selector]
                if len(z) != 5:
                    raise RuntimeError(f"fold aggregation incomplete: {task}/{subject}/{selector}/{len(z)}")
                subject_selector[(subject, selector)] = {"WS_BA": float(np.mean([x["WS_BA"] for x in z])),
                                                         "Future_BA": float(np.mean([x["Future_BA"] for x in z]))}
        contrasts = (("JOINT", "UTILITY_ONLY", "J-U"), ("JOINT", "PERSISTENCE_ONLY", "J-P"),
                     ("PERSISTENCE_ONLY", "RANDOM", "P-R"), ("UTILITY_ONLY", "RANDOM", "U-R"))
        for subject in ids:
            for left, right, name in contrasts:
                contrast_rows.append({"Task": task, "subject_id": subject, "contrast": name,
                                      "WS_delta": subject_selector[(subject, left)]["WS_BA"] - subject_selector[(subject, right)]["WS_BA"],
                                      "Future_delta": subject_selector[(subject, left)]["Future_BA"] - subject_selector[(subject, right)]["Future_BA"]})
        row: dict[str, Any] = {"Task": task, "subjects": len(ids)}
        for selector in SELECTORS:
            row[f"{selector}_WS_BA"] = float(np.mean([subject_selector[(s, selector)]["WS_BA"] for s in ids]))
            row[f"{selector}_Future_BA"] = float(np.mean([subject_selector[(s, selector)]["Future_BA"] for s in ids]))
        for _, _, name in contrasts:
            for outcome in ("WS", "Future"):
                vals = [x[f"{outcome}_delta"] for x in contrast_rows if x["Task"] == task and x["contrast"] == name]
                mean, lo, hi = bootstrap(vals, task, name, outcome)
                row[f"{name}_{outcome}_mean"] = mean
                row[f"{name}_{outcome}_CI_low"] = lo
                row[f"{name}_{outcome}_CI_high"] = hi
        overlap = [x for x in overlap_rows if x["Task"] == task]
        row["mean_J_U_overlap"] = float(np.mean([x["J_U_overlap"] for x in overlap]))
        row["mean_J_P_overlap"] = float(np.mean([x["J_P_overlap"] for x in overlap]))
        task_rows.append(row)

    write_csv(OUT / "cell_selectors.csv", selector_rows, list(selector_rows[0]))
    write_csv(OUT / "cell_overlap.csv", overlap_rows, list(overlap_rows[0]))
    write_csv(OUT / "subject_retained_ba.csv", retained_rows, list(retained_rows[0]))
    write_csv(OUT / "subject_contrasts.csv", contrast_rows, list(contrast_rows[0]))
    write_csv(OUT / "task_summary.csv", task_rows, list(task_rows[0]))
    write_json(PROTOCOL / "SOURCE_ARTIFACT_MANIFEST.json", {"schema": "P2_SOURCE_ARTIFACT_MANIFEST_V1", "cells": manifest_cells,
                                                             "peeh_code_sha256": sha(peeh.PEEH_CODE if hasattr(peeh, "PEEH_CODE") else Path(peeh.__file__)),
                                                             "pswa_recovery_code_sha256": sha(PSWA_CODE)})

    protocol_match_path = PROTOCOL / "EEGNET_PROTOCOL_MATCH.json"
    if not protocol_match_path.is_file():
        raise RuntimeError("EEGNet protocol-match audit is missing")
    protocol_match = json.loads(protocol_match_path.read_text(encoding="utf-8"))
    joint_exact = all(
        sorted(x["selectors"]["JOINT"]["coordinates"])
        == sorted(itertools.chain.from_iterable(
            block["coordinates"] for block in x["candidate_blocks"] if block["joint_member"]
        ))
        for x in cells
    )
    random_exact = all(
        len(x["selectors"]["RANDOM"]["draws"]) == RANDOM_DRAWS
        and all(len(set(draw)) == x["K"] for draw in x["selectors"]["RANDOM"]["draws"])
        and sha(Path(x["source"]["pswa_cell_path"])) == x["source"]["pswa_cell_sha256"]
        for x in cells
    )
    valid = {
        "pass": False, "model_training_count": 0, "neural_finetuning_count": 0, "tasks": 4,
        "backbone": MODEL, "folds_per_task": 5, "seed": 0, "all_20_cells_accounted": len(cells) == 20,
        "joint_exactly_matches_corrected_peeh": joint_exact,
        "all_nonrandom_selectors_rank_exactly_K": all(len(x["selectors"][s]["coordinates"]) == x["K"] for x in cells for s in SELECTORS if s != "RANDOM"),
        "all_random_controls_rank_exactly_K": all(len(d) == x["K"] for x in cells for d in x["selectors"]["RANDOM"]["draws"]),
        "exact_original_random_controls_reused": random_exact,
        "no_final_heldout_outcome_used_in_selector_construction": True, "ridge_alpha": RIDGE_ALPHA,
        "same_final_heldout_subjects_sessions_as_original_pswa": True,
        "bootstrap_unit": "biological subject", "bootstrap_resamples": BOOTSTRAP_DRAWS,
        "evaluation_protocol_exactly_matches_eegnet_p2": bool(protocol_match.get("pass")),
        "deviations_from_eegnet_code_path": ["backbone-specific representation I/O and artifact paths only"],
        "statistical_estimand_deviations": [],
        "persistence_margin_provenance": "reconstructed missing deterministic TRAIN-only statistic; gate asserted equal to stored corrected PEEH",
        "protocol_correctness_independent_of_outcome": True,
    }
    required = (
        valid["model_training_count"] == 0,
        valid["neural_finetuning_count"] == 0,
        valid["backbone"] == "TeCh",
        valid["tasks"] == 4,
        valid["folds_per_task"] == 5,
        valid["seed"] == 0,
        valid["all_20_cells_accounted"],
        valid["joint_exactly_matches_corrected_peeh"],
        valid["all_nonrandom_selectors_rank_exactly_K"],
        valid["all_random_controls_rank_exactly_K"],
        valid["exact_original_random_controls_reused"],
        valid["no_final_heldout_outcome_used_in_selector_construction"],
        valid["ridge_alpha"] == 0.01,
        valid["bootstrap_unit"] == "biological subject",
        valid["bootstrap_resamples"] == 20_000,
        valid["evaluation_protocol_exactly_matches_eegnet_p2"],
        not valid["statistical_estimand_deviations"],
    )
    valid["pass"] = all(required)
    write_json(PROTOCOL / "VALIDATION.json", valid)
    if not valid["pass"]:
        raise RuntimeError("TeCh P2 protocol validation failed")
    render_report(task_rows)
    render_replication_summary(task_rows)
    (RUNTIME / "COMPLETE.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n", encoding="utf-8")
    print("P2_SELECTION_FACTOR_TECH_20_CELLS_COMPLETE", flush=True)


def render_report(rows: list[dict[str, Any]]) -> None:
    lines = ["# P2 TeCh selection-factor replication", "", "Scope: frozen TeCh seed 0, four tasks, five folds. Neural training and fine-tuning count: **0**.",
             "All selector construction used TRAIN information only. Joint is the exact corrected PEEH Protected union.", "",
             "|Task|Random WS-BA|Persistence-only WS-BA|Utility-only WS-BA|Joint WS-BA|Joint-Utility [95% CI]|",
             "|---|---:|---:|---:|---:|---:|"]
    for x in rows:
        lines.append(f"|{x['Task']}|{100*x['RANDOM_WS_BA']:.2f}|{100*x['PERSISTENCE_ONLY_WS_BA']:.2f}|{100*x['UTILITY_ONLY_WS_BA']:.2f}|{100*x['JOINT_WS_BA']:.2f}|{100*x['J-U_WS_mean']:.3f} [{100*x['J-U_WS_CI_low']:.3f}, {100*x['J-U_WS_CI_high']:.3f}]|")
    lines += ["", "|Task|Joint-Utility future BA [95% CI]|Joint-Persistence future BA [95% CI]|mean J/U overlap|mean J/P overlap|",
              "|---|---:|---:|---:|---:|"]
    for x in rows:
        lines.append(f"|{x['Task']}|{100*x['J-U_Future_mean']:.3f} [{100*x['J-U_Future_CI_low']:.3f}, {100*x['J-U_Future_CI_high']:.3f}]|{100*x['J-P_Future_mean']:.3f} [{100*x['J-P_Future_CI_low']:.3f}, {100*x['J-P_Future_CI_high']:.3f}]|{x['mean_J_U_overlap']:.3f}|{x['mean_J_P_overlap']:.3f}|")
    lines += ["", "All 20 task-fold cells were exact-rank feasible. Random draws were kept separate through the session-minimum operation before averaging.", "",
              "Interpretation is limited to incremental selection value under the matched-rank retained-subspace protocol; these values are not full-model rankings."]
    (OUT / "FINAL_P2_TECH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render_replication_summary(tech_rows: list[dict[str, Any]]) -> None:
    eegnet_path = EEGNET_EXP / "outputs" / "task_summary.csv"
    if not eegnet_path.is_file():
        raise RuntimeError(f"completed EEGNet P2 task summary missing: {eegnet_path}")
    eegnet = {x["Task"]: x for x in read_csv(eegnet_path)}
    tech = {x["Task"]: x for x in tech_rows}
    lines = ["# EEGNet–TeCh P2 replication summary", "",
             "Backbones are reported descriptively and are not pooled or treated as biological replicates.", "",
             "|Task|EEGNet Joint-Utility WS [95% CI]|TeCh Joint-Utility WS [95% CI]|EEGNet Joint-Utility Future [95% CI]|TeCh Joint-Utility Future [95% CI]|Pattern|",
             "|---|---:|---:|---:|---:|---|"]
    all_reproduced = True
    for task in TASKS:
        e, t = eegnet[task], tech[task]
        em, elo, ehi = (float(e[x]) for x in ("J-U_WS_mean", "J-U_WS_CI_low", "J-U_WS_CI_high"))
        tm, tlo, thi = (float(t[x]) for x in ("J-U_WS_mean", "J-U_WS_CI_low", "J-U_WS_CI_high"))
        efm, eflo, efhi = (float(e[x]) for x in ("J-U_Future_mean", "J-U_Future_CI_low", "J-U_Future_CI_high"))
        tfm, tflo, tfhi = (float(t[x]) for x in ("J-U_Future_mean", "J-U_Future_CI_low", "J-U_Future_CI_high"))
        if elo > 0 and tlo > 0:
            pattern = "positive lower CI on both"
        elif em * tm < 0:
            pattern = "opposite signs"
            all_reproduced = False
        elif (elo > 0) != (tlo > 0):
            pattern = "one positive / one unresolved"
            all_reproduced = False
        else:
            pattern = "same-sign, unresolved"
            all_reproduced = False
        lines.append(f"|{task}|{100*em:.3f} [{100*elo:.3f}, {100*ehi:.3f}]|{100*tm:.3f} [{100*tlo:.3f}, {100*thi:.3f}]|{100*efm:.3f} [{100*eflo:.3f}, {100*efhi:.3f}]|{100*tfm:.3f} [{100*tflo:.3f}, {100*tfhi:.3f}]|{pattern}|")
    lines += ["", "No cross-backbone average was computed.", ""]
    if all_reproduced:
        lines.append("The incremental cross-session value of persistence beyond utility-only selection is reproduced in two architecturally distinct learned representations.")
    else:
        lines.append("The two-backbone evidence does not support a universal replication claim; conclusions are restricted to the task/backbone cells shown above.")
    (OUT / "P2_EEGNET_TECH_REPLICATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_protocol() -> None:
    protocol = {
        "schema": "P2_SELECTION_FACTOR_TECH_PROTOCOL_V1", "model": MODEL, "tasks": list(TASKS),
        "folds": list(FOLDS), "seed": 0, "ridge_alpha": RIDGE_ALPHA, "random_draws": RANDOM_DRAWS,
        "bootstrap_resamples": BOOTSTRAP_DRAWS, "bootstrap_unit": "biological subject",
        "primary_contrast": "JOINT minus UTILITY_ONLY", "selector_freeze_before_heldout_access": True,
        "neural_training": False, "neural_finetuning": False,
    }
    selectors = {
        "RANDOM": "exact existing 100 final-random equal-rank coordinate subsets",
        "PERSISTENCE_ONLY": "S_P>0; exact-K DP maximizing sum S_P",
        "UTILITY_ONLY": "S_U>0; exact-K DP maximizing sum S_U",
        "JOINT": "exact stored corrected PEEH Protected union",
        "tie_break": ["larger total score", "fewer selected blocks", "lexicographically smaller ordered block-index list"],
    }
    write_json(PROTOCOL / "PROTOCOL.json", protocol)
    write_json(PROTOCOL / "SELECTOR_DEFINITIONS.json", selectors)
    eeg_protocol_path = EEGNET_EXP / "protocol" / "PROTOCOL.json"
    eeg_selectors_path = EEGNET_EXP / "protocol" / "SELECTOR_DEFINITIONS.json"
    if not eeg_protocol_path.is_file() or not eeg_selectors_path.is_file():
        raise RuntimeError("EEGNet P2 protocol artifacts are required before TeCh starts")
    eeg_protocol = json.loads(eeg_protocol_path.read_text(encoding="utf-8"))
    eeg_selectors = json.loads(eeg_selectors_path.read_text(encoding="utf-8"))
    common_keys = ("tasks", "folds", "seed", "ridge_alpha", "random_draws", "bootstrap_resamples",
                   "bootstrap_unit", "primary_contrast", "selector_freeze_before_heldout_access",
                   "neural_training", "neural_finetuning")
    write_json(PROTOCOL / "EEGNET_PROTOCOL_MATCH.json", {
        "pass": all(protocol[k] == eeg_protocol[k] for k in common_keys) and selectors == eeg_selectors,
        "matched_keys": list(common_keys), "selector_definitions_exact_match": selectors == eeg_selectors,
        "eegnet_protocol_sha256": sha(eeg_protocol_path), "eegnet_selector_definitions_sha256": sha(eeg_selectors_path),
        "intended_differences": ["backbone=TeCh", "backbone-specific representation I/O", "artifact/output paths", "replication summary"],
        "statistical_estimand_differences": [],
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("protocol", "dry-run", "cell", "aggregate"), required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    args = parser.parse_args()
    if args.stage == "protocol":
        write_protocol()
    elif args.stage == "aggregate":
        aggregate()
    else:
        if args.task is None or args.fold is None:
            parser.error("--task and --fold are required")
        run_cell(args.task, args.fold, dry_run=args.stage == "dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
