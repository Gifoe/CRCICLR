"""One fixed held-out evaluation after the complete task carrier grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

from task_datasets import (EXP, OUTPUTS, PROTOCOL, TASKS, RawGPUCache, build_model,
                           load_bundle, normalizer, sha256, split_reference, write_json)


def required_lock() -> dict[str, Any]:
    lock = PROTOCOL / "TASK_GENERALITY_PROTOCOL_LOCK.json"
    hashed = PROTOCOL / "TASK_GENERALITY_PROTOCOL_LOCK.sha256"
    if not lock.is_file() or not hashed.is_file() or sha256(lock) != hashed.read_text(encoding="utf-8").strip():
        raise RuntimeError("TASK_GENERALITY_PROTOCOL_INVALID: protocol lock mismatch")
    tests = json.loads((PROTOCOL / "PREFLIGHT_TESTS.json").read_text(encoding="utf-8"))
    if not tests.get("pass") or tests.get("heldout_label_arrays_opened"):
        raise RuntimeError("TASK_GENERALITY_PROTOCOL_INVALID: preflight invalid")
    return json.loads(lock.read_text(encoding="utf-8"))


def provenance() -> dict[tuple[str, int, int, str], dict[str, Any]]:
    p = json.loads((PROTOCOL / "CHECKPOINT_PROVENANCE.json").read_text(encoding="utf-8"))
    rows = p.get("records", [])
    if not p.get("pass") or len(rows) != 60:
        raise RuntimeError("TASK_GENERALITY_PROTOCOL_INVALID: full 60-run carrier grid required")
    by = {(r["task"], int(r["fold"]), int(r["seed"]), r["model"]): r for r in rows}
    if len(by) != 60 or any(not Path(r["checkpoint_path"]).is_file() or sha256(Path(r["checkpoint_path"])) != r["checkpoint_sha256"] for r in rows):
        raise RuntimeError("TASK_GENERALITY_PROTOCOL_INVALID: checkpoint provenance mismatch")
    return by


def score(task: str, y: np.ndarray, logits: np.ndarray) -> dict[str, float | None]:
    pred = logits.argmax(1)
    out: dict[str, float | None] = {"BA": float(balanced_accuracy_score(y, pred)), "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)), "accuracy": float(accuracy_score(y, pred))}
    if task == "ERP":
        prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
        out.update({"AUROC": float(roc_auc_score(y, prob)), "AUPRC": float(average_precision_score(y, prob))})
    else:
        out.update({"AUROC": None, "AUPRC": None})
    return out


def infer(model: torch.nn.Module, cache: RawGPUCache, indices: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            rows.append(model(cache.batch(indices[start:start + 128], mean, std)[0])[0].float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def preflight() -> None:
    required_lock(); by = provenance()
    if not (OUTPUTS / "SEARCH_REPLICATE_RESULTS.csv").is_file():
        raise RuntimeError("TASK_GENERALITY_PROTOCOL_INVALID: SEARCH evaluation must precede heldout evaluation")
    search, heldout, split, _ = split_reference()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schema: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    model_checks = []
    for task, spec in TASKS.items():
        for subject in heldout:
            xp = Path(f"{EXP.parents[1]}")  # keeps this block signal-only; exact path is below
            from task_datasets import task_path
            signal, label = task_path(task, subject, 2, "signal"), task_path(task, subject, 2, "label")
            if not signal.is_file() or not label.is_file():
                raise RuntimeError(f"heldout cache absent: {task}/{subject}")
            x = np.load(signal, mmap_mode="r", allow_pickle=False)
            if x.ndim != 3 or x.shape[1:] != (62, spec["samples"]) or x.dtype != np.float32:
                raise RuntimeError(f"heldout signal schema invalid: {signal}")
            schema[task].append({"subject_id": subject, "signal_path": str(signal), "label_path": str(label), "shape": list(x.shape), "label_opened": False})
        for fold in split["folds"]:
            for seed in (0, 1, 2):
                pair = [by[(task, int(fold["fold_id"]), seed, name)] for name in ("EEGNet", "LiteBN")]
                for record in pair:
                    m = build_model(record["model"], task).to(device)
                    m.load_state_dict(torch.load(record["checkpoint_path"], map_location=device, weights_only=False), strict=True)
                    m.eval()
                    for p in m.parameters(): p.requires_grad_(False)
                    with torch.no_grad(): z, _ = m(torch.zeros((2, 62, spec["samples"]), device=device))
                    if z.shape != (2, spec["classes"]) or m.training or any(p.requires_grad for p in m.parameters()):
                        raise RuntimeError("heldout frozen-model preflight failed")
                    model_checks.append({"task": task, "fold": int(fold["fold_id"]), "seed": seed, "model": record["model"], "frozen": True})
                    del m
    if device.type == "cuda": torch.cuda.empty_cache()
    write_json(PROTOCOL / "HELDOUT_EVALUATION_PREFLIGHT.json", {"pass": True, "heldout_labels_opened": False, "session": 2, "subjects": heldout, "signal_schema": schema, "model_checks": model_checks, "all_60_checkpoint_hashes_verified": True, "no_optimizer": True, "no_training": True, "fixed_logit50": True})
    print("OPENBMI_TASK_HELDOUT_PREFLIGHT_PASS")


def evaluate() -> None:
    required_lock(); by = provenance()
    pf = json.loads((PROTOCOL / "HELDOUT_EVALUATION_PREFLIGHT.json").read_text(encoding="utf-8"))
    if not pf.get("pass") or pf.get("heldout_labels_opened"):
        raise RuntimeError("TASK_GENERALITY_PROTOCOL_INVALID: heldout preflight absent/invalid")
    search, heldout, split, _ = split_reference()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    comp = {task: {k: 0 for k in ("both_correct", "both_wrong", "EEGNet_only_correct", "LiteBN_only_correct", "trials")} for task in TASKS}
    for task in ("ERP", "SSVEP"):
        # This is the sole label-opening path for held-out people.  Session 1 is
        # deliberately absent from this bundle.
        held = load_bundle(task, heldout, sessions=(2,))
        cache = RawGPUCache(held, device)
        source = load_bundle(task, search)
        for fold in split["folds"]:
            mean, std, norm = normalizer(source, fold["inner_train_subjects"])
            for seed in (0, 1, 2):
                models = {}
                for name in ("EEGNet", "LiteBN"):
                    record = by[(task, int(fold["fold_id"]), seed, name)]
                    if record["normalizer"]["mean_std_sha256"] != norm["mean_std_sha256"]:
                        raise RuntimeError("source normalizer provenance mismatch")
                    m = build_model(name, task).to(device)
                    m.load_state_dict(torch.load(record["checkpoint_path"], map_location=device, weights_only=False), strict=True)
                    m.eval()
                    for p in m.parameters(): p.requires_grad_(False)
                    models[name] = m
                for subject in heldout:
                    ix = held.indices([subject], (2,)); y = held.labels(ix)
                    ze = infer(models["EEGNet"], cache, ix, mean, std); zl = infer(models["LiteBN"], cache, ix, mean, std); z50 = (ze + zl) / 2.0
                    for name, logits in (("EEGNet", ze), ("LiteBN", zl), ("LOGIT50", z50)):
                        rows.append({"task": task, "subject_id": subject, "fold": int(fold["fold_id"]), "seed": seed, "method": name, "trials": int(len(y)), **score(task, y, logits)})
                    ep, lp = ze.argmax(1), zl.argmax(1)
                    comp[task]["both_correct"] += int(((ep == y) & (lp == y)).sum())
                    comp[task]["both_wrong"] += int(((ep != y) & (lp != y)).sum())
                    comp[task]["EEGNet_only_correct"] += int(((ep == y) & (lp != y)).sum())
                    comp[task]["LiteBN_only_correct"] += int(((lp == y) & (ep != y)).sum())
                    comp[task]["trials"] += int(len(y))
                del models
                if device.type == "cuda": torch.cuda.empty_cache()
        del source, cache, held
        if device.type == "cuda": torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    if len(frame) != 2 * 14 * 15 * 3 or frame.duplicated(["task", "subject_id", "fold", "seed", "method"]).any():
        raise RuntimeError("TASK_GENERALITY_PROTOCOL_INVALID: heldout replicate cardinality")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUTS / "HELDOUT_REPLICATE_RESULTS.csv", index=False)
    for task, value in comp.items():
        total = value["trials"]
        value.update({f"{k}_fraction": value[k] / total for k in ("both_correct", "both_wrong", "EEGNet_only_correct", "LiteBN_only_correct")})
        value["complementarity_fraction"] = value["EEGNet_only_correct_fraction"] + value["LiteBN_only_correct_fraction"]
        value["description"] = "trial-level descriptive count over all 15 fixed heldout replicates; never used for fusion adaptation"
    write_json(OUTPUTS / "CARRIER_COMPLEMENTARITY.json", comp)
    print("OPENBMI_TASK_HELDOUT_EVALUATION_COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--preflight", action="store_true"); parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.evaluate: parser.error("choose exactly one of --preflight or --evaluate")
    preflight() if args.preflight else evaluate()