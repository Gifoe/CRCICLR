"""Frozen checkpoint-only persistence selector ablation for one cell.

No neural optimizer or training method is imported. Selection uses only TRAIN
source/future representations. Heldout features are inferred after selectors
and rank-matched controls have been frozen for this cell.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
P1 = ROOT.parent
RUNTIME = P1 / "persist_incremental_value_runtime/analysis"
BASELINE_CODE = ROOT / "experiments/persist_eeg_eegconformer_fbcnet_multiseed_v1/code"
PSWA_CODE = ROOT / "experiments/persist_eeg_baseline_metrics_closure_v1/code/published_run_pswa_recovery.py"
os.environ.setdefault("PEEH_REPO", str(ROOT))
os.environ.setdefault("PEEH_RUNTIME", str(P1 / "persist_incremental_value_runtime/seed0_replay"))
os.environ.setdefault("PSWA_RUNTIME", str(RUNTIME))
os.environ.setdefault("SEVEN_RUNTIME", str(P1 / "seven_backbone_fourtask_3seed_runtime"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


pswa = module("incremental_frozen_pswa", PSWA_CODE)
peeh = pswa.peeh
selectors = module("incremental_selectors", Path(__file__).with_name("selectors.py"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gate() -> None:
    source = json.loads((EXP / "protocol/SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    if source["checkpoint_count"] != 420 or source["passed_checkpoint_count"] != 420 or source["problems"]:
        raise RuntimeError("provenance gate failed")
    if os.environ.get("INCREMENTAL_DIRECT_ANALYSIS") == "1":
        amendment = json.loads((EXP / "protocol/DIRECT_ANALYSIS_AMENDMENT.json").read_text(encoding="utf-8"))
        if amendment.get("mode") != "USER_AUTHORIZED_REPLAY_BYPASS" or amendment.get("checkpoint_provenance_required") is not True:
            raise RuntimeError("direct-analysis amendment missing or invalid")
        return
    for name, expected_pass, expected_invalid in (
        ("SEED0_REPLAY_AUDIT.csv", 99, 1), ("SEED0_PSWA_REPLAY_AUDIT.csv", 85, 1)
    ):
        with (EXP / "outputs" / name).open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 100 or any(r["status"] == "FAIL" for r in rows):
            raise RuntimeError(f"seed0 replay gate incomplete: {name}")
        if sum(r["status"] == "PASS" for r in rows) != expected_pass or sum(
            r["status"] == "PROTOCOL_INVALID" for r in rows) != expected_invalid:
            raise RuntimeError(f"seed0 replay count mismatch: {name}")


def audit_row(model: str, task: str, fold: int, seed: int) -> dict[str, str]:
    with (EXP / "protocol/CHECKPOINT_AUDIT.csv").open(newline="", encoding="utf-8") as f:
        matches = [r for r in csv.DictReader(f) if (r["model"], r["task"], int(r["fold"]), int(r["seed"])) ==
                   (model, task, fold, seed)]
    if len(matches) != 1 or matches[0]["status"] != "PASS":
        raise RuntimeError("checkpoint provenance not passed")
    return matches[0]


def build_model(model: str, record: dict, checkpoint: Path, device: torch.device):
    task = record["task"]
    row = {"Model": model, "Task": task, "fold": record["fold"], "seed": record["seed"],
           "channels": int(record.get("channels") or (58 if task == "WBCIC_MI" else 62)),
           "samples": int(record.get("samples") or (250 if task == "OpenBMI_ERP" else 1000)),
           "classes": int(record["classes"]), "checkpoint_path": str(checkpoint),
           "recipe_name": record.get("recipe", {}).get("name"),
           "trainable_parameters": int(record.get("trainable_parameters", record.get("parameters", 0)))}
    if model in peeh.MODELS:
        net, head = peeh.build_model(row, device)
        return net, head
    if str(BASELINE_CODE) not in sys.path:
        sys.path.insert(0, str(BASELINE_CODE))
    baseline_models = module("incremental_new_baseline_models", BASELINE_CODE / "models.py")
    net = baseline_models.build_model(model, row["channels"], row["samples"], row["classes"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["state_dict"], strict=True)
    net.eval().to(device)
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    head = net.classifier if model == "EEGConformer" else net.head
    return net, head


def representation(net, head, x: np.ndarray, model: str, device: torch.device) -> np.ndarray:
    if model in peeh.MODELS:
        return peeh.representations(net, head, x, model, device, batch=32)
    filterbank = module("incremental_fixed_filterbank", BASELINE_CODE / "filterbank.py") if model == "FBCNet" else None
    parts = []
    captured = []
    handle = head.register_forward_pre_hook(lambda _module, arguments: captured.append(arguments[0].detach()))
    try:
        with torch.inference_mode():
            for start in range(0, len(x), 32):
                value = np.ascontiguousarray(x[start:start + 32], dtype=np.float32)
                if filterbank is not None:
                    value = filterbank.transform(value)
                captured.clear()
                net(torch.from_numpy(value).to(device))
                if len(captured) != 1:
                    raise RuntimeError("classifier-input hook did not fire exactly once")
                parts.append(captured[0].reshape(len(value), -1).float().cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(parts).astype(np.float32, copy=False)


def ba_by_subject(pred: np.ndarray, labels: np.ndarray, owner: np.ndarray) -> dict[str, float]:
    return {subject: float(balanced_accuracy_score(labels[owner == subject], pred[owner == subject]))
            for subject in peeh.natural_subjects(owner)}


def result_path(model: str, task: str, fold: int, seed: int) -> Path:
    return RUNTIME / "cells" / model.lower() / task.lower() / f"fold{fold}_seed{seed}.json"


def run(model: str, task: str, fold: int, seed: int) -> None:
    target = result_path(model, task, fold, seed)
    if target.is_file():
        found = json.loads(target.read_text(encoding="utf-8"))
        if found["identity"] != [model, task, fold, seed]:
            raise RuntimeError("stale cell cache identity")
        print("CELL_CACHED", model, task, fold, seed, found["status"], flush=True)
        return
    audit = audit_row(model, task, fold, seed)
    checkpoint = Path(audit["checkpoint_path"])
    if digest(checkpoint) != audit["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed since provenance lock")
    record = json.loads((checkpoint.parent / "record.json").read_text(encoding="utf-8"))
    data = pswa.load_arrays(task, fold)
    if data["normalizer"]["mean_std_sha256"] != audit["normalizer_sha256"]:
        raise RuntimeError("TRAIN normalizer hash mismatch")
    capped = pswa.cap_data(data, task, fold)
    expected = set(OPENBMI_HELD if task != "WBCIC_MI" else WBCIC_HELD)
    if any(set(value["subjects"].astype(str)) != expected for value in capped["evaluation"].values()):
        raise RuntimeError("heldout cohort mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, head = build_model(model, record, checkpoint, device)
    hs = representation(net, head, capped["source_x"], model, device)
    hf = representation(net, head, capped["future_x"], model, device)
    if model == "EEGConformer":
        expected_dim = int(net.classifier[0].in_features)
    else:
        expected_dim = int(head.in_features)
    if hs.shape[1] != expected_dim or hf.shape[1] != expected_dim:
        raise RuntimeError("classifier input representation dimension mismatch")
    h = np.concatenate([hs, hf])
    y = np.concatenate([capped["source_y"], capped["future_y"]])
    owner = np.concatenate([capped["source_subjects"], capped["future_subjects"]])
    session = np.concatenate([np.full(len(hs), data["source_session"]),
                              np.full(len(hf), data["future_session"])]).astype(np.int64)
    try:
        spec = peeh.spectrum(h, y, owner, session, task, model, fold)
    except RuntimeError as error:
        if "active rank below 4" not in str(error):
            raise
        value = {"identity": [model, task, fold, seed], "status": "PROTOCOL_INVALID_ACTIVE_RANK", "reason": str(error),
                 "checkpoint_sha256": audit["checkpoint_sha256"], "training_performed": False}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        print("CELL_INVALID", model, task, fold, seed, flush=True)
        return
    spec["classes"] = data["classes"]
    pu_dims, assignments = peeh.select_protected(h, y, owner, session, spec, model, task, fold)
    selected = selectors.choose(spec, assignments)
    if tuple(pu_dims) != selected["PU"].coordinates:
        raise RuntimeError("PU is not the unchanged formal Protected union")
    result = {"identity": [model, task, fold, seed], "status": "EMPTY_PU" if selected["PU_empty"] else "ESTIMABLE",
              "checkpoint_sha256": audit["checkpoint_sha256"], "normalizer_sha256": audit["normalizer_sha256"],
              "active_rank": spec["rank"], "representation_dim": hs.shape[1], "k": selected["k"],
              "blocks": spec["blocks"], "support": spec["support"], "utility_evidence": assignments,
              "selector_blocks": {name: list(selected[name].block_ids) if selected[name] else [] for name in ("PU", "U_only", "P_only")},
              "selector_coordinates": {name: list(selected[name].coordinates) if selected[name] else [] for name in ("PU", "U_only", "P_only")},
              "utility_scores": selected["utility_scores"], "persistence_scores": selected["persistence_scores"],
              "training_performed": False, "selector_uses_heldout": False, "random_controls_shared": True,
              "subject_results": [], "session_results": []}
    if selected["PU_empty"]:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("CELL_EMPTY", model, task, fold, seed, flush=True)
        return
    k = selected["k"]
    controls = [np.random.default_rng(peeh.stable_seed("final-random", model, task, fold, draw)).choice(
        np.arange(spec["rank"]), k, replace=False).astype(int).tolist() for draw in range(100)]
    if any(len(set(c)) != k for c in controls):
        raise RuntimeError("random exact-rank check failed")
    result["random_controls"] = controls
    q_source = peeh.canonical(hs, spec).astype(np.float32)
    retained = {name: {} for name in ("PU", "U_only", "P_only")}
    random_retained = {draw: {} for draw in range(100)}
    eval_representations = {}
    for session_name, value in capped["evaluation"].items():
        he = representation(net, head, value["x"], model, device)
        if he.shape[1] != expected_dim:
            raise RuntimeError("heldout representation dimension mismatch")
        eval_representations[session_name] = he
        q_eval = peeh.canonical(he, spec).astype(np.float32)
        labels = value["y"]
        subjects = value["subjects"].astype(str)
        for name in retained:
            dims = list(selected[name].coordinates)
            pred = pswa.ridge_scores(q_source[:, dims], capped["source_y"], q_eval[:, dims], data["classes"]).argmax(1)
            retained[name][session_name] = ba_by_subject(pred, labels, subjects)
        for draw, dims in enumerate(controls):
            pred = pswa.ridge_scores(q_source[:, dims], capped["source_y"], q_eval[:, dims], data["classes"]).argmax(1)
            random_retained[draw][session_name] = ba_by_subject(pred, labels, subjects)
    future_key = f"S{data['future_session']}"
    if future_key not in eval_representations:
        raise RuntimeError("future evaluation session missing")
    future = capped["evaluation"][future_key]
    he = eval_representations[future_key]
    q_eval = peeh.canonical(he, spec).astype(np.float32)
    erasure = peeh.FixedStandardizerKernelRidge(hs, capped["source_y"], he, data["classes"], spec,
                                                 qfit=q_source, qeval=q_eval, raw_base=peeh._erasure_base(spec))
    erased = {}
    for name in retained:
        erased[name] = ba_by_subject(erasure.predict(selected[name].coordinates)[0], future["y"], future["subjects"].astype(str))
    random_erased = {draw: ba_by_subject(erasure.predict(dims)[0], future["y"], future["subjects"].astype(str))
                     for draw, dims in enumerate(controls)}
    for subject in peeh.natural_subjects(future["subjects"]):
        random_worst = float(np.mean([min(random_retained[draw][session][subject] for session in eval_representations)
                                      for draw in range(100)]))
        random_erased_mean = float(np.mean([random_erased[draw][subject] for draw in range(100)]))
        for name in retained:
            future_ba = retained[name][future_key][subject]
            worst = min(retained[name][session][subject] for session in eval_representations)
            result["subject_results"].append({"subject_id": subject, "selector": name, "future_BA": future_ba,
                                              "worst_BA": worst, "random_worst_BA": random_worst,
                                              "PSWA_pp": 100 * (worst - random_worst),
                                              "PEEH_pp": 100 * (random_erased_mean - erased[name][subject])})
            for session_name in eval_representations:
                result["session_results"].append({"subject_id": subject, "selector": name,
                                                  "session": session_name, "BA": retained[name][session_name][subject]})
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.part")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    print("CELL_COMPLETE", model, task, fold, seed, "k", k, flush=True)


OPENBMI_HELD = ("4", "12", "13", "17", "18", "24", "25", "29", "36", "37", "39", "42", "51", "54")
WBCIC_HELD = ("sub-4", "sub-8", "sub-10", "sub-15", "sub-20", "sub-39", "sub-40", "sub-43", "sub-46", "sub-51")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer", "EEGConformer", "FBCNet"))
    parser.add_argument("task", choices=("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI"))
    parser.add_argument("fold", type=int, choices=range(5))
    parser.add_argument("seed", type=int, choices=range(3))
    args = parser.parse_args()
    gate()
    run(args.model, args.task, args.fold, args.seed)


if __name__ == "__main__":
    main()
