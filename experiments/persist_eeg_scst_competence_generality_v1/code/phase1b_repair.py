from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, log_loss


REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
EXP = REPO / "experiments" / "persist_eeg_scst_competence_generality_v1"
PREV_CODE = REPO / "experiments" / "persist_eeg_fm_rescue_stage0" / "code"
sys.path.insert(0, str(PREV_CODE))
import common as c


CONFIGS = (
    {"id": "R1_lr1e-5", "encoder_lr": 1e-5, "downstream_lr": 3e-4, "weight_decay": 1e-2},
    {"id": "R1_lr3e-5", "encoder_lr": 3e-5, "downstream_lr": 3e-4, "weight_decay": 1e-2},
)
THRESHOLDS = {"OpenBMI": 0.7519166667, "WBCIC": 0.7684300821}
FOLDS = range(5); SEEDS = range(3); MAX_EPOCHS = 25; MIN_EPOCHS = 10; PATIENCE = 6; BATCH_SIZE = 96


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def mean_subject_ba(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> float:
    pred = logits.argmax(1); values = []
    for subject in c.subject_sort(np.unique(subjects.astype(str))):
        mask = subjects.astype(str) == subject; values.append(balanced_accuracy_score(labels[mask], pred[mask]))
    return float(np.mean(values))


def nll(labels: np.ndarray, logits: np.ndarray) -> float:
    value = logits.astype(np.float64); value -= value.max(1, keepdims=True); probability = np.exp(value); probability /= probability.sum(1, keepdims=True)
    return float(log_loss(labels, probability, labels=[0, 1]))


def repair_path(dataset: str, config: str, fold: int, seed: int) -> Path:
    return EXP / "runtime" / "cbramod_repair" / dataset / config / f"fold-{fold}" / f"seed-{seed}.pt"


def configure_trainable(model: c.FMTask) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    for parameter in model.parameters(): parameter.requires_grad_(False)
    last = model.encoder.encoder.layers[-1]
    for parameter in last.parameters(): parameter.requires_grad_(True)
    for parameter in model.task_projector.parameters(): parameter.requires_grad_(True)
    for parameter in model.head.parameters(): parameter.requires_grad_(True)
    return list(last.parameters()), list(model.task_projector.parameters()) + list(model.head.parameters())


def train_one(dataset: str, config: dict[str, object], fold: int, seed: int, output: Path, device: torch.device) -> dict[str, object]:
    if output.is_file():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        if payload.get("complete") is True: return payload["record"]
    data = c.load_data(dataset); role = c.fold_roles(dataset, fold); sessions = c.SOURCE_SESSIONS[dataset]
    train_idx = c.row_indices(data.metadata, role["model_fit"], sessions); val_idx = c.row_indices(data.metadata, role["validation"], sessions); labels = data.metadata.label.to_numpy(np.int64); xall = c.input_array(dataset)
    set_seed(stable_seed("repair", dataset, config["id"], fold, seed)); model = c.load_anchor("CBraMod", dataset, fold, seed, device); encoder_parameters, downstream_parameters = configure_trainable(model)
    optimizer = torch.optim.AdamW([{"params": encoder_parameters, "lr": float(config["encoder_lr"])}, {"params": downstream_parameters, "lr": float(config["downstream_lr"])}], weight_decay=float(config["weight_decay"]))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1); rng = np.random.default_rng(stable_seed("repair-order", dataset, config["id"], fold, seed)); best_key = (-math.inf, -math.inf); best_state = None; best_epoch = 0; stale = 0; history = []
    for epoch in range(MAX_EPOCHS):
        model.train(); order = rng.permutation(train_idx); losses = []; cosine = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * epoch / MAX_EPOCHS))
        optimizer.param_groups[0]["lr"] = float(config["encoder_lr"]) * cosine; optimizer.param_groups[1]["lr"] = float(config["downstream_lr"]) * cosine
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]; x = torch.from_numpy(np.asarray(xall[idx], np.float32)).to(device).reshape(len(idx), len(c.channels(dataset)), 4, 200); y = torch.as_tensor(labels[idx], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(x); loss = criterion(logits, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(encoder_parameters + downstream_parameters, 3.0); optimizer.step(); losses.append(float(loss.detach()))
        validation = c.infer(model, dataset, val_idx, device); ba = mean_subject_ba(validation["labels"], validation["logits"], validation["subjects"]); validation_nll = nll(validation["labels"], validation["logits"]); key = (ba, -validation_nll)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_BA": ba, "validation_NLL": validation_nll})
        if key > best_key:
            best_key = key; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best_epoch = epoch + 1; stale = 0
        else: stale += 1
        print(f"[repair] {dataset} {config['id']} fold={fold} seed={seed} epoch={epoch+1} valBA={ba:.5f} best={best_key[0]:.5f}", flush=True)
        if epoch + 1 >= MIN_EPOCHS and stale >= PATIENCE: break
    record = {"dataset": dataset, "config": config["id"], "fold": fold, "seed": seed, "validation_BA": float(best_key[0]), "validation_NLL": float(-best_key[1]), "best_epoch": best_epoch, "last_encoder_block_trainable": True, "task_projector_trainable": True, "head_trainable": True, "future_session_used": False, "history": history}
    output.parent.mkdir(parents=True, exist_ok=True); temp = output.with_suffix(".pt.part"); torch.save({"complete": True, "state_dict": best_state, "config": config, "record": record}, temp); os.replace(temp, output); del model; torch.cuda.empty_cache(); return record


def load_repaired(dataset: str, fold: int, seed: int, path: Path, device: torch.device) -> c.FMTask:
    payload = torch.load(path, map_location="cpu", weights_only=False); model = c.FMTask("CBraMod", dataset, stable_seed("head", "CBraMod", dataset, fold, seed)); model.load_state_dict(payload["state_dict"]); return model.to(device).eval()


def save_rep(path: Path, value: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(".npz.part")
    clean = {key: (np.asarray(item).astype("U") if np.asarray(item).dtype == object else item) for key, item in value.items()}
    with temp.open("wb") as stream: np.savez_compressed(stream, **clean)
    os.replace(temp, path)


def main() -> None:
    device = torch.device("cuda"); rows = []; selections = []; task_rows = []
    for dataset in c.DATASETS:
        for config in CONFIGS:
            for fold in FOLDS: rows.append(train_one(dataset, config, fold, 0, repair_path(dataset, config["id"], fold, 0), device))
        grid = pd.DataFrame([r for r in rows if r["dataset"] == dataset and r["seed"] == 0]).groupby("config", as_index=False).agg(mean_validation_BA=("validation_BA", "mean"), mean_validation_NLL=("validation_NLL", "mean"), minimum_fold_BA=("validation_BA", "min"))
        selected = grid.sort_values(["mean_validation_BA", "mean_validation_NLL", "minimum_fold_BA", "config"], ascending=[False, True, False, True]).iloc[0]; config = next(value for value in CONFIGS if value["id"] == selected["config"]); selections.append({"dataset": dataset, **selected.to_dict(), **config})
        for fold in FOLDS:
            for seed in SEEDS:
                path = repair_path(dataset, config["id"], fold, seed); record = train_one(dataset, config, fold, seed, path, device)
                if not any(r["dataset"] == dataset and r["config"] == config["id"] and r["fold"] == fold and r["seed"] == seed for r in rows): rows.append(record)
                model = load_repaired(dataset, fold, seed, path, device); data = c.load_data(dataset); role = c.fold_roles(dataset, fold)
                partitions = {}
                for part in ("model_fit", "validation", "outcome"):
                    idx = c.row_indices(data.metadata, role[part], c.SOURCE_SESSIONS[dataset]); rep = c.infer(model, dataset, idx, device); partitions[part] = rep; save_rep(EXP / "runtime" / "cbramod_repaired_representations" / dataset / f"fold-{fold}" / f"seed-{seed}" / f"{part}.npz", rep)
                outcome = partitions["outcome"]; ba = mean_subject_ba(outcome["labels"], outcome["logits"], outcome["subjects"])
                task_rows.append({"model": "CBraMod-R1", "type": "FM", "dataset": dataset, "fold": fold, "seed": seed, "BA": ba, "NLL": nll(outcome["labels"], outcome["logits"]), "threshold": THRESHOLDS[dataset], "representation_changed": True, "future_session_used": False})
                print(f"[repair-outcome] {dataset} fold={fold} seed={seed} BA={ba:.5f}", flush=True); del model; torch.cuda.empty_cache()
    pd.DataFrame(rows).drop(columns=["history"], errors="ignore").to_csv(EXP / "runtime" / "CBRAMOD_REPAIR_SOURCE_VALIDATION.csv", index=False); pd.DataFrame(selections).to_csv(EXP / "results" / "CBRAMOD_REPAIR_SELECTION.csv", index=False); task = pd.DataFrame(task_rows); task.to_csv(EXP / "results" / "CBRAMOD_REPAIR_PER_FOLD.csv", index=False)
    summary = task.groupby(["model", "type", "dataset"], as_index=False).agg(BA=("BA", "mean"), NLL=("NLL", "mean"), threshold=("threshold", "first"), folds=("fold", "nunique"), seeds=("seed", "nunique")); summary["competent"] = summary.BA >= summary.threshold; summary.to_csv(EXP / "results" / "CBRAMOD_REPAIR_COMPETENCE.csv", index=False)
    terminal = "CBRAMOD_REPAIR_COMPETENT" if summary.competent.all() else "CBRAMOD_COMPETENCE_NOT_RECOVERED"
    result = {"schema": "CBRAMOD_LIMITED_REPRESENTATION_REPAIR_V1", "family": "R1_last_encoder_block_plus_task_projector_and_head", "selection": selections, "datasets": summary.to_dict("records"), "terminal": terminal, "requires_complete_scst_reaudit": True, "future_session_used": False}
    (EXP / "results" / "CBRAMOD_REPAIR_RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    ledger = EXP / "COMPETENCE_ITERATION_LEDGER.md"; text = ledger.read_text(encoding="utf-8"); text += "\n## Iteration 2 — limited R1 representation repair\n\n- Trigger: Phase 1A failed both frozen competence thresholds.\n- Change: fine-tune only CBraMod encoder layer 11, official task projector, and task head.\n- Search: encoder LR {1e-5, 3e-5}; downstream LR 3e-4; weight decay 1e-2.\n- Selection: source validation BA, NLL tie-break.\n- SCST utility available: no.\n- Keep/reject: selected once per dataset; full admissibility re-audit is mandatory.\n"; ledger.write_text(text, encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
