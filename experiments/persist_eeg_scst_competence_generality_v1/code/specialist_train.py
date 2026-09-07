from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from specialist_models import build_model, count_parameters


REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
EXP = REPO / "experiments" / "persist_eeg_scst_competence_generality_v1"
RUNTIME = EXP / "runtime"; RESULTS = EXP / "results"
MODELS = ("FBCNet", "ATCNet", "EEGInceptionMI")
DATASETS = ("OpenBMI", "WBCIC"); FOLDS = range(5); SEEDS = range(3)
SOURCE_SESSIONS = {"OpenBMI": (1, 2), "WBCIC": (0, 1)}
THRESHOLDS = {"OpenBMI": 0.7519166667, "WBCIC": 0.7684300821}
CONFIGS = {
    model: (
        {"id": "recipe_a", "lr": 1e-3, "weight_decay": 1e-3, "max_epochs": 30, "batch_size": 48 if model == "FBCNet" else 128, "model": {}},
        {"id": "recipe_b", "lr": 3e-4, "weight_decay": 1e-2, "max_epochs": 36, "batch_size": 48 if model == "FBCNet" else 128, "model": {}},
    ) for model in MODELS
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


P2 = load_module("scst_gen_p2", REPO / "experiments" / "persist_eeg_subject_invariance_stress_test_v1" / "code" / "common.py")
P3 = load_module("scst_gen_p3", REPO / "experiments" / "persist_eeg_wbcic_independent_replication_v1" / "code" / "common.py")


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_data(dataset: str):
    d = P2.load_data() if dataset == "OpenBMI" else P3.load_data(); meta = d.metadata.copy()
    meta["subject_id"] = meta.subject_id.astype(str).str.replace("sub-", "", regex=False); meta["session_id"] = meta.session_id.astype(int); meta["label"] = meta.label.astype(int)
    return d.x, meta.reset_index(drop=True)


def subject_sort(values) -> list[str]:
    return sorted((str(v).replace("sub-", "") for v in values), key=lambda x: (int(x) if x.isdigit() else 10**9, x))


def roles(dataset: str, fold: int) -> dict[str, tuple[str, ...]]:
    row = P2.frozen_fold(fold) if dataset == "OpenBMI" else P3.frozen_fold(fold)
    if dataset == "OpenBMI": value = {"model_fit": row["inner_train"], "validation": row["inner_validation"], "outcome": row["outcome"]}
    else: value = {"model_fit": row["model_fit"], "validation": row["validation_discovery"], "outcome": row["outcome"]}
    return {key: tuple(subject_sort(items)) for key, items in value.items()}


def indices(meta: pd.DataFrame, subjects, sessions) -> np.ndarray:
    mask = meta.subject_id.astype(str).isin(set(map(str, subjects))).to_numpy(copy=True); mask &= meta.session_id.isin(set(map(int, sessions))).to_numpy(); return np.flatnonzero(mask).astype(np.int64)


def normalize(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, np.float32); value = value - value.mean(-1, keepdims=True); scale = value.std(-1, keepdims=True); return value / np.maximum(scale, 1e-6)


def subject_ba(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> float:
    pred = logits.argmax(1); return float(np.mean([balanced_accuracy_score(labels[subjects.astype(str) == subject], pred[subjects.astype(str) == subject]) for subject in subject_sort(np.unique(subjects.astype(str)))]))


def metrics(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    pred = logits.argmax(1); z = logits.astype(np.float64); z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return {"BA": subject_ba(labels, logits, subjects), "trial_BA": float(balanced_accuracy_score(labels, pred)), "macro_F1": float(f1_score(labels, pred, average="macro")), "NLL": float(log_loss(labels, p, labels=[0, 1]))}


@torch.no_grad()
def infer(model: nn.Module, raw: np.ndarray, meta: pd.DataFrame, selected: np.ndarray, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    model.eval(); logits = []; features = []
    for start in range(0, len(selected), batch_size):
        idx = selected[start:start + batch_size]; x = torch.from_numpy(normalize(raw[idx])).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16): h = model.forward_features(x); z = model.head(h)
        features.append(h.float().cpu().numpy()); logits.append(z.float().cpu().numpy())
    picked = meta.iloc[selected]
    return {"indices": selected, "features": np.concatenate(features).astype(np.float32), "logits": np.concatenate(logits).astype(np.float32), "labels": picked.label.to_numpy(np.int64), "subjects": picked.subject_id.astype(str).to_numpy(), "sessions": picked.session_id.to_numpy(np.int64)}


def checkpoint_path(model: str, dataset: str, config: str, fold: int, seed: int) -> Path:
    return RUNTIME / "specialist_checkpoints" / model / dataset / config / f"fold-{fold}" / f"seed-{seed}.pt"


def train(model_name: str, dataset: str, config: dict[str, object], fold: int, seed: int, output: Path, device: torch.device) -> dict[str, object]:
    if output.is_file():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        if payload.get("complete") is True: return payload["record"]
    raw, meta = load_data(dataset); role = roles(dataset, fold); sessions = SOURCE_SESSIONS[dataset]
    train_idx = indices(meta, role["model_fit"], sessions); val_idx = indices(meta, role["validation"], sessions); channels = raw.shape[1]
    set_seed(stable_seed("specialist", model_name, dataset, config["id"], fold, seed)); model = build_model(model_name, channels, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])); criterion = nn.CrossEntropyLoss(); max_epochs = int(config["max_epochs"]); batch_size = int(config["batch_size"])
    labels = meta.label.to_numpy(np.int64); best_key = (-math.inf, -math.inf); best_state = None; best_epoch = 0; stale = 0; history = []; rng = np.random.default_rng(stable_seed("order", model_name, dataset, config["id"], fold, seed))
    for epoch in range(max_epochs):
        model.train(); order = rng.permutation(train_idx); losses = []; cosine = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
        for group in optimizer.param_groups: group["lr"] = float(config["lr"]) * cosine
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]; x = torch.from_numpy(normalize(raw[idx])).to(device); y = torch.as_tensor(labels[idx], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(x); loss = criterion(logits, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); optimizer.step(); losses.append(float(loss.detach()))
        val = infer(model, raw, meta, val_idx, batch_size, device); score = metrics(val["labels"], val["logits"], val["subjects"]); key = (score["BA"], -score["NLL"])
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_BA": score["BA"], "validation_NLL": score["NLL"]})
        if key > best_key:
            best_key = key; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best_epoch = epoch + 1; stale = 0
        else: stale += 1
        print(f"[specialist] {model_name} {dataset} {config['id']} fold={fold} seed={seed} epoch={epoch+1} valBA={score['BA']:.5f} best={best_key[0]:.5f}", flush=True)
        if epoch + 1 >= 12 and stale >= 7: break
    record = {"model": model_name, "dataset": dataset, "config": config["id"], "fold": fold, "seed": seed, "validation_BA": float(best_key[0]), "validation_NLL": float(-best_key[1]), "best_epoch": best_epoch, "parameters": count_parameters(model), "train_rows": len(train_idx), "validation_rows": len(val_idx), "future_session_used": False, "history": history}
    output.parent.mkdir(parents=True, exist_ok=True); temp = output.with_suffix(".pt.part"); torch.save({"complete": True, "state_dict": best_state, "config": config, "record": record}, temp); os.replace(temp, output); del model; torch.cuda.empty_cache(); return record


def load_checkpoint(model_name: str, dataset: str, path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False); raw, _ = load_data(dataset); model = build_model(model_name, raw.shape[1], payload["config"]); model.load_state_dict(payload["state_dict"]); return model.to(device).eval(), payload


def save_rep(path: Path, value: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(".npz.part")
    clean = {key: (np.asarray(item).astype("U") if np.asarray(item).dtype == object else item) for key, item in value.items()}
    with temp.open("wb") as stream: np.savez_compressed(stream, **clean)
    os.replace(temp, path)


def main() -> None:
    device = torch.device("cuda"); training_rows = []; selection_rows = []; task_rows = []
    for model_name in MODELS:
        for dataset in DATASETS:
            for config in CONFIGS[model_name]:
                for fold in FOLDS:
                    training_rows.append(train(model_name, dataset, config, fold, 0, checkpoint_path(model_name, dataset, config["id"], fold, 0), device))
            grid = pd.DataFrame([row for row in training_rows if row["model"] == model_name and row["dataset"] == dataset and row["seed"] == 0]).groupby("config", as_index=False).agg(mean_validation_BA=("validation_BA", "mean"), mean_validation_NLL=("validation_NLL", "mean"), minimum_fold_BA=("validation_BA", "min"))
            selected = grid.sort_values(["mean_validation_BA", "mean_validation_NLL", "minimum_fold_BA", "config"], ascending=[False, True, False, True]).iloc[0]; selected_config = next(value for value in CONFIGS[model_name] if value["id"] == selected["config"])
            selection_rows.append({"model": model_name, "dataset": dataset, **selected.to_dict(), "selected_before_outcome": True})
            for fold in FOLDS:
                for seed in SEEDS:
                    path = checkpoint_path(model_name, dataset, selected_config["id"], fold, seed); record = train(model_name, dataset, selected_config, fold, seed, path, device)
                    if not any(row["model"] == model_name and row["dataset"] == dataset and row["fold"] == fold and row["seed"] == seed and row["config"] == selected_config["id"] for row in training_rows): training_rows.append(record)
                    model, payload = load_checkpoint(model_name, dataset, path, device); raw, meta = load_data(dataset); role = roles(dataset, fold); batch = int(selected_config["batch_size"])
                    partitions = {}
                    for part in ("model_fit", "validation", "outcome"):
                        selected_idx = indices(meta, role[part], SOURCE_SESSIONS[dataset]); rep = infer(model, raw, meta, selected_idx, batch, device); partitions[part] = rep; save_rep(RUNTIME / "specialist_representations" / model_name / dataset / f"fold-{fold}" / f"seed-{seed}" / f"{part}.npz", rep)
                    score = metrics(partitions["outcome"]["labels"], partitions["outcome"]["logits"], partitions["outcome"]["subjects"])
                    task_rows.append({"model": model_name, "type": "Specialist", "dataset": dataset, "fold": fold, "seed": seed, "config": selected_config["id"], "BA": score["BA"], "macro_F1": score["macro_F1"], "NLL": score["NLL"], "threshold": THRESHOLDS[dataset], "future_session_used": False, "parameters": record["parameters"]})
                    print(f"[specialist-outcome] {model_name} {dataset} fold={fold} seed={seed} BA={score['BA']:.5f}", flush=True); del model; torch.cuda.empty_cache()
            pd.DataFrame(training_rows).drop(columns=["history"], errors="ignore").to_csv(RUNTIME / "SPECIALIST_SOURCE_VALIDATION_RUNS.csv", index=False)
            pd.DataFrame(selection_rows).to_csv(RESULTS / "SPECIALIST_SELECTION.csv", index=False)
            pd.DataFrame(task_rows).to_csv(RESULTS / "SPECIALIST_TASK_PER_FOLD.csv", index=False)
    tasks = pd.DataFrame(task_rows); summary = tasks.groupby(["model", "type", "dataset"], as_index=False).agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), NLL=("NLL", "mean"), threshold=("threshold", "first"), folds=("fold", "nunique"), seeds=("seed", "nunique"), parameters=("parameters", "first")); summary["competent"] = summary.BA >= summary.threshold
    summary.to_csv(RESULTS / "SPECIALIST_SCREEN.csv", index=False)
    lines = ["# Specialist training ledger", "", "Selection used source model-fit/validation only; future sessions were not accessed.", ""]
    for row in selection_rows:
        lines += [f"## {row['model']} — {row['dataset']}", "", f"- Implementation: frozen clean-room canonical family in `code/specialist_models.py`.", f"- Parameter count: {int(tasks[(tasks.model == row['model']) & (tasks.dataset == row['dataset'])].parameters.iloc[0])}.", f"- Grid: recipe_a (lr=1e-3, wd=1e-3, 30 epochs) versus recipe_b (lr=3e-4, wd=1e-2, 36 epochs).", f"- Selected: {row['config']} using mean source-validation BA={row['mean_validation_BA']:.6f}; NLL={row['mean_validation_NLL']:.6f}.", ""]
    (EXP / "SPECIALIST_TRAINING_LEDGER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
