"""Frozen matched heterogeneous-vs-homogeneous ensemble control.

Preflight intentionally touches only checkpoints, source signal normalizers,
and held-out signal schemas.  Evaluation has one label-opening path and keeps
the canonical trial identity `(block, subject, future_session, trial_index)`
attached to every prediction in memory before any fusion is formed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

REPO = Path(os.environ.get("MATCHED_ENSEMBLE_REPO", Path(__file__).resolve().parents[3])).resolve()
EXP = REPO / "experiments" / "persist_eeg_matched_homogeneous_ensemble_control_v1"
OUT = EXP / "outputs"
CODE = EXP / "code"
FREEZE = EXP / "PROTOCOL_FREEZE.md"
FREEZE_HASH = EXP / "PROTOCOL_FREEZE.sha256"
SEED_PAIRS = ((0, 1), (0, 2), (1, 2))
BOOTSTRAP_SEED, BOOTSTRAP_N = 20260909, 10_000
BLOCKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")

MI_CODE = REPO / "experiments" / "persist_eeg_final_heldout_confirmation_v1" / "code"
TASK_CODE = REPO / "experiments" / "persist_eeg_openbmi_task_generality_v1" / "code"
for source in (str(MI_CODE), str(TASK_CODE)):
    if source not in sys.path:
        sys.path.insert(0, source)

from load_final_carriers import (  # noqa: E402
    CHANNELS, FUTURE_SESSION, audit_checkpoints, carrier_folds, checkpoint_records,
    holdout_memberships, load_eval_subject, load_frozen_model, normalizer_provenance,
    normalize_to_device, sha256, signal_schema,
)
from task_datasets import (  # noqa: E402
    TASKS, RawGPUCache, build_model, load_bundle, normalizer, split_reference,
    task_path,
)


def clean(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return None


def freeze_digest() -> str:
    if not FREEZE.is_file():
        raise RuntimeError(f"missing immutable protocol freeze: {FREEZE}")
    return sha256(FREEZE)


def code_digest() -> str:
    return sha256(Path(__file__))


def require_freeze() -> str:
    actual = freeze_digest()
    if not FREEZE_HASH.is_file() or FREEZE_HASH.read_text(encoding="utf-8").strip() != actual:
        raise RuntimeError("MATCHED_ENSEMBLE_PROTOCOL_INVALID: protocol freeze hash mismatch")
    return actual


def metric(y: np.ndarray, logits: np.ndarray, rule: str) -> tuple[float, float, np.ndarray]:
    if rule == "LOGIT50":
        prediction = logits.argmax(axis=1)
    elif rule == "PROB50":
        # callers pass already averaged probabilities in the PROB50 path.
        prediction = logits.argmax(axis=1)
    else:
        raise ValueError(rule)
    return (float(balanced_accuracy_score(y, prediction)),
            float(f1_score(y, prediction, average="macro", zero_division=0)), prediction)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fuse(a: np.ndarray, b: np.ndarray, rule: str) -> np.ndarray:
    if rule == "LOGIT50":
        return 0.5 * a + 0.5 * b
    if rule == "PROB50":
        return 0.5 * softmax(a) + 0.5 * softmax(b)
    raise ValueError(rule)


def infer_mi(model: torch.nn.Module, x: np.ndarray, mean: np.ndarray, std: np.ndarray,
             device: torch.device) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 128):
            z, _ = model(normalize_to_device(x[start:start + 128], mean, std, device))
            chunks.append(z.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def infer_task(model: torch.nn.Module, cache: RawGPUCache, indices: np.ndarray,
               mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            z, _ = model(cache.batch(indices[start:start + 128], mean, std)[0])
            chunks.append(z.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def task_records() -> dict[tuple[str, int, int, str], dict[str, Any]]:
    protocol = REPO / "experiments" / "persist_eeg_openbmi_task_generality_v1" / "protocol" / "CHECKPOINT_PROVENANCE.json"
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    expected = {(task, fold, seed, name) for task in ("ERP", "SSVEP") for fold in range(5)
                for seed in range(3) for name in ("EEGNet", "LiteBN")}
    by = {(str(r["task"]), int(r["fold"]), int(r["seed"]), str(r["model"])): r for r in records}
    if not payload.get("pass") or set(by) != expected:
        raise RuntimeError("MATCHED_ENSEMBLE_PROTOCOL_INVALID: incomplete task checkpoint provenance")
    for key, row in by.items():
        path = Path(row["checkpoint_path"])
        if not path.is_file() or sha256(path) != row["checkpoint_sha256"]:
            raise RuntimeError(f"MATCHED_ENSEMBLE_PROTOCOL_INVALID: task checkpoint mismatch {key}")
    return by


def mi_records() -> dict[tuple[str, int, int, str], dict[str, Any]]:
    records = audit_checkpoints(checkpoint_records())
    by = {(str(r["dataset"]), int(r["fold"]), int(r["seed"]), str(r["model"])): r for r in records}
    expected = {(dataset, fold, seed, name) for dataset in ("OpenBMI", "WBCIC") for fold in range(5)
                for seed in range(3) for name in ("EEGNet", "LiteBN")}
    if set(by) != expected or not all(r["sha_match"] for r in by.values()):
        raise RuntimeError("MATCHED_ENSEMBLE_PROTOCOL_INVALID: incomplete MI checkpoint provenance")
    return by


def _model_check_mi(records: dict[tuple[str, int, int, str], dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    checks = []
    for (dataset, fold, seed, name), record in sorted(records.items()):
        model = load_frozen_model(name, CHANNELS[dataset], Path(record["checkpoint_path"]), device)
        with torch.no_grad():
            logits, _ = model(torch.zeros((2, CHANNELS[dataset], 1000), device=device))
        if logits.shape != (2, 2) or model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError(f"MI frozen model smoke check failed: {dataset}/{fold}/{seed}/{name}")
        checks.append({"block": f"{dataset}_MI", "fold": fold, "seed": seed, "model": name,
                       "frozen": True, "logits_shape": list(logits.shape)})
        del model
    return checks


def _model_check_task(records: dict[tuple[str, int, int, str], dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    checks = []
    for (task, fold, seed, name), record in sorted(records.items()):
        model = build_model(name, task).to(device)
        model.load_state_dict(torch.load(record["checkpoint_path"], map_location=device, weights_only=False), strict=True)
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        spec = TASKS[task]
        with torch.no_grad():
            logits, _ = model(torch.zeros((2, 62, spec["samples"]), device=device))
        if logits.shape != (2, spec["classes"]) or model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError(f"task frozen model smoke check failed: {task}/{fold}/{seed}/{name}")
        checks.append({"block": f"OpenBMI_{task}", "fold": fold, "seed": seed, "model": name,
                       "frozen": True, "logits_shape": list(logits.shape)})
        del model
    return checks


def preflight() -> None:
    """Metadata/signal/checkpoint-only step; never reads held-out label values."""
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = freeze_digest()
    mi = mi_records(); task = task_records()
    memberships, _ = holdout_memberships()
    task_search, task_heldout, task_split, _ = split_reference()
    if memberships["OpenBMI"] != task_heldout:
        raise RuntimeError("MATCHED_ENSEMBLE_PROTOCOL_INVALID: task/MI OpenBMI heldout mismatch")
    folds = carrier_folds()
    if any(len(folds[d]) != 5 for d in ("OpenBMI", "WBCIC")) or len(task_split["folds"]) != 5:
        raise RuntimeError("MATCHED_ENSEMBLE_PROTOCOL_INVALID: incomplete frozen folds")
    normalizers, _ = normalizer_provenance()
    mi_schema = {d: {s: signal_schema(d, s, FUTURE_SESSION[d]) for s in memberships[d]} for d in ("OpenBMI", "WBCIC")}
    task_schema: dict[str, dict[str, Any]] = {}
    for task_name, spec in TASKS.items():
        task_schema[task_name] = {}
        for subject in task_heldout:
            signal, label = task_path(task_name, subject, 2, "signal"), task_path(task_name, subject, 2, "label")
            if not signal.is_file() or not label.is_file():
                raise FileNotFoundError(f"task cache absent: {task_name}/{subject}")
            array = np.load(signal, mmap_mode="r", allow_pickle=False)
            if array.ndim != 3 or array.shape[1:] != (62, spec["samples"]) or array.dtype != np.float32:
                raise RuntimeError(f"task signal schema mismatch: {signal}")
            task_schema[task_name][subject] = {"signal_path": str(signal), "label_path": str(label),
                                               "shape": list(array.shape), "label_opened": False}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checks = _model_check_mi(mi, device) + _model_check_task(task, device)
    if device.type == "cuda": torch.cuda.empty_cache()
    provenance = {
        "source_branch": "codex/persist-eeg-openbmi-task-generality-v1",
        "analysis_git_sha_at_preflight": git_sha(),
        "training_executed": False,
        "inference_only": True,
        "fusion_weights": {"LOGIT50": [0.5, 0.5], "PROB50_sensitivity": [0.5, 0.5]},
        "MI_records": list(mi.values()),
        "task_records": list(task.values()),
        "checkpoint_counts": {"MI": len(mi), "OpenBMI_ERP_SSVEP": len(task), "total": len(mi) + len(task)},
        "future_session": {"OpenBMI_MI": 2, "OpenBMI_ERP": 2, "OpenBMI_SSVEP": 2, "WBCIC_MI": 2},
    }
    alignment = {
        "phase": "preflight", "pass": True, "heldout_label_values_opened": False,
        "canonical_trial_identity": ["dataset", "task", "subject_id", "future_session", "trial_index"],
        "class_order": {"OpenBMI_MI": [0, 1], "WBCIC_MI": [0, 1], "OpenBMI_ERP": [0, 1], "OpenBMI_SSVEP": [0, 1, 2, 3]},
        "subject_membership": {"OpenBMI_MI": memberships["OpenBMI"], "WBCIC_MI": memberships["WBCIC"],
                               "OpenBMI_ERP": task_heldout, "OpenBMI_SSVEP": task_heldout},
        "signal_schema": {"MI": mi_schema, "TASK": task_schema},
        "required_pairs": [list(p) for p in SEED_PAIRS], "required_folds": [0, 1, 2, 3, 4],
        "checks": checks,
    }
    dump_json(EXP / "ARTIFACT_PROVENANCE.json", provenance)
    dump_json(EXP / "ALIGNMENT_AUDIT.json", alignment)
    dump_json(EXP / "PREFLIGHT_TESTS.json", {
        "pass": True, "protocol_sha256": freeze, "heldout_label_values_opened": False,
        "implementation_sha256": code_digest(),
        "all_frozen_checkpoints_hash_verified": True, "all_models_eval_only": True,
        "all_three_seeds": True, "all_five_folds": True, "all_seed_pairs_locked": True,
        "no_training": True, "no_calibration": True, "no_adaptive_fusion": True,
        "source_only_MI_normalizers_verified": True, "source_task_search_subjects": task_search,
        "model_checks": len(checks),
    })
    FREEZE_HASH.write_text(freeze + "\n", encoding="utf-8")
    print("MATCHED_ENSEMBLE_PREFLIGHT_PASS", flush=True)


def _record_subject(block: str, dataset: str, task: str, subject: str, fold: int, i: int, j: int,
                    family: str, rule: str, y: np.ndarray, a: np.ndarray, b: np.ndarray,
                    identity: np.ndarray) -> dict[str, Any]:
    ba_a, f1_a, p_a = metric(y, a, "LOGIT50")
    ba_b, f1_b, p_b = metric(y, b, "LOGIT50")
    combined = fuse(a, b, rule)
    ba, f1, prediction = metric(y, combined, rule)
    if len(identity) != len(y) or len(np.unique(identity)) != len(identity):
        raise RuntimeError(f"alignment failure while forming {block}/{subject}/f{fold}/{i}{j}")
    classes = list(range(int(np.max(y)) + 1))
    count = [int((y == c).sum()) for c in classes]
    correct_a = [int(((p_a == y) & (y == c)).sum()) for c in classes]
    correct_b = [int(((p_b == y) & (y == c)).sum()) for c in classes]
    correct_ensemble = [int(((prediction == y) & (y == c)).sum()) for c in classes]
    return {
        "block": block, "dataset": dataset, "task": task, "subject_id": str(subject), "future_session": 2,
        "fold": fold, "seed_i": i, "seed_j": j, "family": family, "fusion_rule": rule,
        "trials": int(len(y)), "trial_identity_sha256": hashlib.sha256("\n".join(identity.tolist()).encode()).hexdigest(),
        "label_sha256": hashlib.sha256(np.asarray(y, dtype=np.int64).tobytes()).hexdigest(),
        "class_order": json.dumps(list(range(int(np.max(y)) + 1))),
        "BA_A": ba_a, "BA_B": ba_b, "macro_F1_A": f1_a, "macro_F1_B": f1_b,
        "ensemble_BA": ba, "ensemble_macro_F1": f1,
        "gain_over_best_pp": 100.0 * (ba - max(ba_a, ba_b)),
        "gain_over_mean_pp": 100.0 * (ba - 0.5 * (ba_a + ba_b)),
        "exclusive_correct_fraction": float(np.mean((p_a == y) != (p_b == y))),
        "correct_A": int((p_a == y).sum()), "correct_B": int((p_b == y).sum()),
        "ensemble_correct": int((prediction == y).sum()),
        "class_counts": json.dumps(count), "class_correct_A": json.dumps(correct_a),
        "class_correct_B": json.dumps(correct_b), "class_correct_ensemble": json.dumps(correct_ensemble),
    }


def _aggregate_pair_subjects(rows: list[dict[str, Any]], block: str, dataset: str, task: str,
                             fold: int, i: int, j: int, rule: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[row["subject_id"]].append(row)
    output: list[dict[str, Any]] = []
    for subject, orient in groups.items():
        if {x["family"] for x in orient} != {"Heterogeneous_A", "Heterogeneous_B"} or len(orient) != 2:
            raise RuntimeError("heterogeneous orientation cardinality failure")
        a, b = sorted(orient, key=lambda x: x["family"])
        merged = {**a, "family": "Heterogeneous_orientation_mean", "orientation_count": 2,
                  "ensemble_BA": float(np.mean([a["ensemble_BA"], b["ensemble_BA"]])),
                  "ensemble_macro_F1": float(np.mean([a["ensemble_macro_F1"], b["ensemble_macro_F1"]])),
                  "gain_over_best_pp": float(np.mean([a["gain_over_best_pp"], b["gain_over_best_pp"]])),
                  "gain_over_mean_pp": float(np.mean([a["gain_over_mean_pp"], b["gain_over_mean_pp"]])),
                  "exclusive_correct_fraction": float(np.mean([a["exclusive_correct_fraction"], b["exclusive_correct_fraction"]])),
                  "orientation_BA_difference_pp": 100.0 * (a["ensemble_BA"] - b["ensemble_BA"])}
        output.append(merged)
    return output


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Balanced accuracy after pooling all subject trials in a fixed pair unit."""
    parsed = [tuple(np.asarray(json.loads(r[key]), dtype=float) for key in
                    ("class_counts", "class_correct_A", "class_correct_B", "class_correct_ensemble")) for r in rows]
    counts = np.sum([x[0] for x in parsed], axis=0)
    if np.any(counts <= 0):
        raise RuntimeError("aggregate BA has an absent class")
    ba_a = float(np.mean(np.sum([x[1] for x in parsed], axis=0) / counts))
    ba_b = float(np.mean(np.sum([x[2] for x in parsed], axis=0) / counts))
    ba_ensemble = float(np.mean(np.sum([x[3] for x in parsed], axis=0) / counts))
    return {"aggregate_BA_A": ba_a, "aggregate_BA_B": ba_b, "aggregate_BA": ba_ensemble,
            "aggregate_G_best_pp": 100.0 * (ba_ensemble - max(ba_a, ba_b)),
            "aggregate_G_mean_pp": 100.0 * (ba_ensemble - 0.5 * (ba_a + ba_b))}


def run_block(block: str, dataset: str, task: str,
              payload: dict[int, dict[int, dict[str, dict[str, np.ndarray]]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate all pair families from complete aligned logits for one block."""
    subject_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for fold in range(5):
        for i, j in SEED_PAIRS:
            base = payload[fold]
            per_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for subject in sorted(base[i]["EEGNet"], key=lambda x: int(str(x).replace("sub-", ""))):
                e_i, e_j = base[i]["EEGNet"][subject]["logits"], base[j]["EEGNet"][subject]["logits"]
                l_i, l_j = base[i]["LiteBN"][subject]["logits"], base[j]["LiteBN"][subject]["logits"]
                y, identity = base[i]["EEGNet"][subject]["y"], base[i]["EEGNet"][subject]["identity"]
                for seed in (i, j):
                    for name in ("EEGNet", "LiteBN"):
                        other = base[seed][name][subject]
                        if not np.array_equal(y, other["y"]) or not np.array_equal(identity, other["identity"]):
                            raise RuntimeError(f"non-provable one-to-one trial alignment: {block} f{fold} pair{i}{j} {subject}")
                for rule in ("LOGIT50", "PROB50"):
                    records = [
                        _record_subject(block, dataset, task, subject, fold, i, j, "EEGNet_homogeneous", rule, y, e_i, e_j, identity),
                        _record_subject(block, dataset, task, subject, fold, i, j, "LiteBN_homogeneous", rule, y, l_i, l_j, identity),
                        _record_subject(block, dataset, task, subject, fold, i, j, "Heterogeneous_A", rule, y, e_i, l_j, identity),
                        _record_subject(block, dataset, task, subject, fold, i, j, "Heterogeneous_B", rule, y, e_j, l_i, identity),
                    ]
                    subject_rows.extend(records)
                    for record in records: per_family[f"{rule}:{record['family']}"] .append(record)
            for rule in ("LOGIT50", "PROB50"):
                hetero = _aggregate_pair_subjects(per_family[f"{rule}:Heterogeneous_A"] + per_family[f"{rule}:Heterogeneous_B"], block, dataset, task, fold, i, j, rule)
                subject_rows.extend(hetero)
                ee, ll = per_family[f"{rule}:EEGNet_homogeneous"], per_family[f"{rule}:LiteBN_homogeneous"]
                by_subject = {r["subject_id"]: r for r in hetero}
                for subject in by_subject:
                    h = by_subject[subject]
                    e = next(r for r in ee if r["subject_id"] == subject)
                    l = next(r for r in ll if r["subject_id"] == subject)
                    subject_rows.append({**h, "family": "Matched_contrast", "G_EE_pp": e["gain_over_best_pp"],
                                         "G_LL_pp": l["gain_over_best_pp"], "G_EL_pp": h["gain_over_best_pp"],
                                         "D_avg_pp": h["gain_over_best_pp"] - 0.5 * (e["gain_over_best_pp"] + l["gain_over_best_pp"]),
                                         "D_best_pp": h["gain_over_best_pp"] - max(e["gain_over_best_pp"], l["gain_over_best_pp"])})
                for family, values in (("EEGNet_homogeneous", ee), ("LiteBN_homogeneous", ll), ("Heterogeneous_orientation_mean", hetero)):
                    if family == "Heterogeneous_orientation_mean":
                        a_metrics = aggregate_metrics(per_family[f"{rule}:Heterogeneous_A"])
                        b_metrics = aggregate_metrics(per_family[f"{rule}:Heterogeneous_B"])
                        aggregate = {key: float(np.mean([a_metrics[key], b_metrics[key]])) for key in a_metrics}
                    else:
                        aggregate = aggregate_metrics(values)
                    pair_rows.append({"block": block, "dataset": dataset, "task": task, "fold": fold, "seed_i": i, "seed_j": j,
                                      "family": family, "fusion_rule": rule, "n_subjects": len(values),
                                      "mean_subject_BA": float(np.mean([r["ensemble_BA"] for r in values])),
                                      "aggregate_mean_subject_macro_F1": float(np.mean([r["ensemble_macro_F1"] for r in values])),
                                      "G_best_pp": float(np.mean([r["gain_over_best_pp"] for r in values])),
                                      "G_mean_pp": float(np.mean([r["gain_over_mean_pp"] for r in values])),
                                      "exclusive_correct_fraction": float(np.mean([r["exclusive_correct_fraction"] for r in values])),
                                      **aggregate})
                hh = [r for r in subject_rows if r["block"] == block and r["fold"] == fold and r["seed_i"] == i and r["seed_j"] == j and r["family"] == "Matched_contrast" and r["fusion_rule"] == rule]
                pair_rows.append({"block": block, "dataset": dataset, "task": task, "fold": fold, "seed_i": i, "seed_j": j,
                                  "family": "Matched_contrast", "fusion_rule": rule, "n_subjects": len(hh),
                                  "D_avg_pp": float(np.mean([r["D_avg_pp"] for r in hh])),
                                  "D_best_pp": float(np.mean([r["D_best_pp"] for r in hh]))})
    return subject_rows, pair_rows


def bootstrap(values: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))].mean(axis=1)
    return {"mean_pp": float(values.mean()), "median_pp": float(np.median(values)),
            "ci_low_pp": float(np.quantile(draws, 0.025)), "ci_high_pp": float(np.quantile(draws, 0.975)),
            "positive_subjects": int((values > 0).sum()), "positive_fraction": float((values > 0).mean()),
            "harm_le_minus_1pp": int((values <= -1.0).sum()), "harm_le_minus_3pp": int((values <= -3.0).sum()),
            "harm_le_minus_5pp": int((values <= -5.0).sum()), "n_subjects": int(len(values))}


def summarize(subject: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary = subject[subject.fusion_rule.eq("LOGIT50")]
    for block in BLOCKS:
        frame = primary[primary.block.eq(block)]
        for family, measure in (("EEGNet_homogeneous", "gain_over_best_pp"), ("LiteBN_homogeneous", "gain_over_best_pp"),
                                ("Heterogeneous_orientation_mean", "gain_over_best_pp"), ("Matched_contrast", "D_avg_pp"),
                                ("Matched_contrast", "D_best_pp")):
            values = frame[frame.family.eq(family)].groupby("subject_id", sort=True)[measure].mean().to_numpy()
            result = bootstrap(values, rng)
            rows.append({"block": block, "family": family, "measure": measure, "fusion_rule": "LOGIT50", **result})
        for family in ("EEGNet_homogeneous", "LiteBN_homogeneous", "Heterogeneous_orientation_mean"):
            values = frame[frame.family.eq(family)].groupby("subject_id", sort=True)["exclusive_correct_fraction"].mean().to_numpy()
            rows.append({"block": block, "family": family, "measure": "exclusive_correct_fraction", "fusion_rule": "LOGIT50", **bootstrap(100.0 * values, rng)})
    summary = pd.DataFrame(rows)
    block_values = {}
    for measure in ("D_avg_pp", "D_best_pp"):
        per_block = []
        for block in BLOCKS:
            vals = primary[(primary.block.eq(block)) & (primary.family.eq("Matched_contrast"))].groupby("subject_id", sort=True)[measure].mean().to_numpy()
            block_values[block + ":" + measure] = vals
            per_block.append(vals)
        draws = np.zeros(BOOTSTRAP_N, dtype=float)
        for b, vals in enumerate(per_block):
            draws += vals[rng.integers(0, len(vals), size=(BOOTSTRAP_N, len(vals)))].mean(axis=1) / len(per_block)
        means = [float(x.mean()) for x in per_block]
        block_values["global:" + measure] = {"mean_pp": float(np.mean(means)), "ci_low_pp": float(np.quantile(draws, .025)),
                                                "ci_high_pp": float(np.quantile(draws, .975)), "block_means_pp": dict(zip(BLOCKS, means))}
    d_avg, d_best = block_values["global:D_avg_pp"], block_values["global:D_best_pp"]
    positive_blocks = sum(x > 0 for x in d_avg["block_means_pp"].values())
    large_reproducible_disadvantage = False
    for block in BLOCKS:
        row = summary[(summary.block.eq(block)) & (summary.family.eq("Matched_contrast")) & (summary.measure.eq("D_avg_pp"))].iloc[0]
        large_reproducible_disadvantage |= bool(row.mean_pp <= -1.0 and row.ci_high_pp < 0)
    if d_avg["mean_pp"] > 0 and d_avg["ci_low_pp"] > 0 and positive_blocks >= 3 and not large_reproducible_disadvantage:
        terminal = "HETEROGENEOUS_COMPLEMENTARITY_SUPPORTED"
    elif d_avg["mean_pp"] > 0:
        terminal = "HETEROGENEOUS_ADVANTAGE_WEAK"
    elif sum(x <= 0 for x in d_avg["block_means_pp"].values()) >= 3:
        terminal = "HOMOGENEOUS_ENSEMBLING_MATCHES_OR_EXCEEDS"
    else:
        terminal = "GENERIC_ENSEMBLING_NOT_RULED_OUT"
    sensitivity = []
    for rule in ("LOGIT50", "PROB50"):
        for block in BLOCKS:
            f = subject[(subject.block.eq(block)) & (subject.fusion_rule.eq(rule)) & (subject.family.eq("Matched_contrast"))]
            sensitivity.append({"block": block, "fusion_rule": rule,
                                "D_avg_pp": float(f.groupby("subject_id")["D_avg_pp"].mean().mean()),
                                "D_best_pp": float(f.groupby("subject_id")["D_best_pp"].mean().mean())})
    global_summary = {"bootstrap": {"unit": "subject", "replicates": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
                                     "global_method": "stratified within-block subject bootstrap; equal mean of four blocks"},
                      "global_LOGIT50": {"D_avg": d_avg, "D_best": d_best, "positive_blocks_D_avg": positive_blocks,
                                         "large_reproducible_disadvantage": large_reproducible_disadvantage},
                      "terminal": terminal, "PROB50_sensitivity": sensitivity}
    comp = summary[summary.measure.eq("exclusive_correct_fraction")].copy()
    return summary, global_summary, comp


def write_decision(summary: pd.DataFrame, global_summary: dict[str, Any], comp: pd.DataFrame) -> None:
    g = global_summary["global_LOGIT50"]
    lines = ["# Matched homogeneous-ensemble control decision", "",
             f"Terminal: `{global_summary['terminal']}`.", "",
             "| Block | EEG homo G_best | LiteBN homo G_best | Hetero G_best | D_avg | D_best | D_avg 95% CI |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for block in BLOCKS:
        def find(family: str, measure: str) -> pd.Series:
            return summary[(summary.block.eq(block)) & (summary.family.eq(family)) & (summary.measure.eq(measure))].iloc[0]
        ee, ll, he = find("EEGNet_homogeneous", "gain_over_best_pp"), find("LiteBN_homogeneous", "gain_over_best_pp"), find("Heterogeneous_orientation_mean", "gain_over_best_pp")
        da, db = find("Matched_contrast", "D_avg_pp"), find("Matched_contrast", "D_best_pp")
        lines.append(f"| {block} | {ee.mean_pp:+.3f} | {ll.mean_pp:+.3f} | {he.mean_pp:+.3f} | {da.mean_pp:+.3f} | {db.mean_pp:+.3f} | [{da.ci_low_pp:+.3f}, {da.ci_high_pp:+.3f}] |")
    lines += ["", f"Equal-block global `D_avg`: **{g['D_avg']['mean_pp']:+.3f} pp** "
              f"(95% CI [{g['D_avg']['ci_low_pp']:+.3f}, {g['D_avg']['ci_high_pp']:+.3f}]).",
              f"Equal-block global `D_best`: **{g['D_best']['mean_pp']:+.3f} pp** "
              f"(95% CI [{g['D_best']['ci_low_pp']:+.3f}, {g['D_best']['ci_high_pp']:+.3f}]).",
              "", "All five folds, all three unordered seed pairs, and both heterogeneous orientations were retained. No model was trained in this control.",
              "The correct interpretation is limited to whether the frozen heterogeneous 50/50 fusion has added value beyond these matched ordinary two-instance ensemble controls; it does not establish distinct neural mechanisms."]
    (EXP / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate() -> None:
    require_freeze()
    pf = json.loads((EXP / "PREFLIGHT_TESTS.json").read_text(encoding="utf-8"))
    if not pf.get("pass") or pf.get("heldout_label_values_opened") or pf.get("implementation_sha256") != code_digest():
        raise RuntimeError("MATCHED_ENSEMBLE_PROTOCOL_INVALID: metadata-only preflight required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mi, task = mi_records(), task_records()
    memberships, _ = holdout_memberships()
    _, normalizers = normalizer_provenance()
    all_subject_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    audit_blocks: dict[str, Any] = {}
    # MI blocks: held-out labels open only here, then all frozen models infer.
    for dataset in ("OpenBMI", "WBCIC"):
        block = f"{dataset}_MI"; held = {s: load_eval_subject(dataset, s) for s in memberships[dataset]}
        payload: dict[int, dict[int, dict[str, dict[str, dict[str, np.ndarray]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for fold in range(5):
            mean, std = normalizers[f"{dataset}:{fold}"]
            for seed in range(3):
                for name in ("EEGNet", "LiteBN"):
                    record = mi[(dataset, fold, seed, name)]
                    model = load_frozen_model(name, CHANNELS[dataset], Path(record["checkpoint_path"]), device)
                    for subject, (x, y) in held.items():
                        identity = np.asarray([f"{dataset}|MI|{subject}|{FUTURE_SESSION[dataset]}|{k}" for k in range(len(y))])
                        payload[fold][seed][name][subject] = {"logits": infer_mi(model, x, mean, std, device), "y": y, "identity": identity}
                    del model
                if device.type == "cuda": torch.cuda.empty_cache()
        srows, prows = run_block(block, dataset, "MI", payload)
        all_subject_rows += srows; all_pair_rows += prows
        audit_blocks[block] = {"subjects": memberships[dataset], "future_session": FUTURE_SESSION[dataset], "trial_identity_rule": "dataset|task|subject|session|row_index", "all_models_and_seeds_exactly_aligned": True}
    # OpenBMI ERP/SSVEP blocks: shared cache contains only the fixed future session for held-out people.
    search, heldout, split, _ = split_reference()
    for task_name in ("ERP", "SSVEP"):
        block = f"OpenBMI_{task_name}"; held = load_bundle(task_name, heldout, sessions=(2,)); source = load_bundle(task_name, search)
        cache = RawGPUCache(held, device)
        payload = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for fold_record in split["folds"]:
            fold = int(fold_record["fold_id"]); mean, std, norm = normalizer(source, fold_record["inner_train_subjects"])
            for seed in range(3):
                for name in ("EEGNet", "LiteBN"):
                    record = task[(task_name, fold, seed, name)]
                    if record["normalizer"]["mean_std_sha256"] != norm["mean_std_sha256"]:
                        raise RuntimeError(f"source normalizer provenance mismatch: {task_name}/f{fold}/s{seed}/{name}")
                    model = build_model(name, task_name).to(device)
                    model.load_state_dict(torch.load(record["checkpoint_path"], map_location=device, weights_only=False), strict=True)
                    model.eval()
                    for parameter in model.parameters(): parameter.requires_grad_(False)
                    for subject in heldout:
                        indices = held.indices([subject], (2,)); y = held.labels(indices)
                        identity = np.asarray([f"OpenBMI|{task_name}|{subject}|2|{held.rows[int(ix)].index}" for ix in indices])
                        payload[fold][seed][name][subject] = {"logits": infer_task(model, cache, indices, mean, std), "y": y, "identity": identity}
                    del model
                if device.type == "cuda": torch.cuda.empty_cache()
        del cache, source, held
        if device.type == "cuda": torch.cuda.empty_cache()
        srows, prows = run_block(block, "OpenBMI", task_name, payload)
        all_subject_rows += srows; all_pair_rows += prows
        audit_blocks[block] = {"subjects": heldout, "future_session": 2, "trial_identity_rule": "dataset|task|subject|session|source_row_index", "all_models_and_seeds_exactly_aligned": True}
    subject = pd.DataFrame(all_subject_rows); pairs = pd.DataFrame(all_pair_rows)
    required_subject = {"EEGNet_homogeneous", "LiteBN_homogeneous", "Heterogeneous_A", "Heterogeneous_B", "Heterogeneous_orientation_mean", "Matched_contrast"}
    if set(subject.block.unique()) != set(BLOCKS) or not required_subject.issubset(set(subject.family)):
        raise RuntimeError("MATCHED_ENSEMBLE_PROTOCOL_INVALID: incomplete result matrix")
    if subject.duplicated(["block", "subject_id", "fold", "seed_i", "seed_j", "family", "fusion_rule"]).any():
        raise RuntimeError("MATCHED_ENSEMBLE_PROTOCOL_INVALID: duplicate subject result")
    subject.to_csv(EXP / "subject_level_results.csv", index=False)
    pairs.to_csv(EXP / "pair_level_results.csv", index=False)
    summary, global_summary, comp = summarize(subject)
    summary.to_csv(EXP / "block_summary.csv", index=False)
    comp.to_csv(EXP / "complementarity_summary.csv", index=False)
    dump_json(EXP / "global_summary.json", global_summary)
    audit = json.loads((EXP / "ALIGNMENT_AUDIT.json").read_text(encoding="utf-8"))
    audit.update({"phase": "evaluation", "pass": True, "heldout_label_values_opened": True,
                  "all_blocks": audit_blocks, "no_rows_dropped": True, "no_duplicate_trials": True,
                  "all_checkpoint_hashes_verified_before_inference": True,
                  "all_pairs": [list(x) for x in SEED_PAIRS], "both_heterogeneous_orientations_retained": True,
                  "no_orientation_selection": True, "fusion_weight_exactly": 0.5})
    dump_json(EXP / "ALIGNMENT_AUDIT.json", audit)
    write_decision(summary, global_summary, comp)
    from plot_matched_ensemble_control import plot
    plot(EXP / "pair_level_results.csv", EXP / "matched_homogeneous_ensemble_control.png")
    print("MATCHED_ENSEMBLE_EVALUATION_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    preflight() if args.preflight else evaluate()


if __name__ == "__main__":
    main()
