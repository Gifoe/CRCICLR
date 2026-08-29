from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, log_loss


REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
EXP = REPO / "experiments" / "persist_eeg_scst_competence_generality_v1"
PREV_REP = REPO / "experiments" / "persist_eeg_fm_rescue_stage0" / "runtime" / "representations" / "CBraMod"
THRESHOLDS = {"OpenBMI": 0.7519166667, "WBCIC": 0.7684300821}
FOLDS = range(5)
SEEDS = range(3)


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_part(dataset: str, fold: int, seed: int, part: str) -> dict[str, np.ndarray]:
    path = PREV_REP / dataset / f"fold-{fold}" / f"seed-{seed}" / f"{part}.npz"
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


class Head(nn.Module):
    def __init__(self, name: str, dropout: float):
        super().__init__()
        if name == "H0": layers: list[nn.Module] = [nn.Linear(200, 2)]
        elif name == "H1": layers = [nn.Linear(200, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 2)]
        elif name == "H2": layers = [nn.Linear(200, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2)]
        else: raise KeyError(name)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def subject_ba(y: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> float:
    pred = logits.argmax(1); values = []
    for subject in sorted(np.unique(subjects.astype(str))):
        mask = subjects.astype(str) == subject
        values.append(balanced_accuracy_score(y[mask], pred[mask]))
    return float(np.mean(values))


def nll(y: np.ndarray, logits: np.ndarray) -> float:
    z = logits.astype(np.float64); z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return float(log_loss(y, p, labels=[0, 1]))


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(0, dtype=np.float64).astype(np.float32)
    scale = train.std(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-5)
    return (train - mean) / scale, (other - mean) / scale, mean, scale


def fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    architecture: str,
    lr: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    subjects_val: np.ndarray | None = None,
    fixed_epochs: int | None = None,
) -> tuple[Head, dict[str, object]]:
    set_seed(seed); device = torch.device("cuda")
    model = Head(architecture, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    xt = torch.from_numpy(x_train).float().to(device); yt = torch.from_numpy(y_train).long().to(device)
    counts = pd.Series(subjects_train.astype(str)).value_counts()
    weights = np.asarray([1.0 / counts[str(s)] for s in subjects_train], np.float32)
    weights /= weights.mean(); wt = torch.from_numpy(weights).to(device)
    xv = torch.from_numpy(x_val).float().to(device) if x_val is not None else None
    epochs = int(fixed_epochs or 50); best_key = (-math.inf, -math.inf); best_state = None; best_epoch = 0; stale = 0
    rng = np.random.default_rng(seed); batch_size = min(512, len(x_train))
    history = []
    for epoch in range(epochs):
        model.train(); order = rng.permutation(len(x_train)); losses = []
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xt[idx]); raw = nn.functional.cross_entropy(logits, yt[idx], reduction="none")
            loss = (raw * wt[idx]).mean(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        if xv is None:
            history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))}); continue
        model.eval()
        with torch.no_grad(): val_logits = model(xv).float().cpu().numpy()
        ba = subject_ba(y_val, val_logits, subjects_val); val_nll = nll(y_val, val_logits); key = (ba, -val_nll)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_BA": ba, "validation_NLL": val_nll})
        if key > best_key:
            best_key = key; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best_epoch = epoch + 1; stale = 0
        else: stale += 1
        if epoch + 1 >= 12 and stale >= 8: break
    if xv is not None:
        assert best_state is not None
        model.load_state_dict(best_state)
    return model, {"best_epoch": int(best_epoch or epochs), "history": history, "validation_BA": float(best_key[0]) if xv is not None else None, "validation_NLL": float(-best_key[1]) if xv is not None else None}


def recipes() -> list[dict[str, object]]:
    rows = []
    for architecture in ("H0", "H1", "H2"):
        drops = (0.0,) if architecture == "H0" else (0.0, 0.2, 0.4)
        for lr in (3e-4, 1e-3, 3e-3):
            for wd in (1e-4, 1e-3, 1e-2):
                for dropout in drops:
                    rows.append({"architecture": architecture, "lr": lr, "weight_decay": wd, "dropout": dropout})
    return rows


def main() -> None:
    output_rows = []; grid_rows = []; selections = []
    for dataset in ("OpenBMI", "WBCIC"):
        for rid, recipe in enumerate(recipes()):
            fold_rows = []
            for fold in FOLDS:
                train = load_part(dataset, fold, 0, "model_fit"); val = load_part(dataset, fold, 0, "validation")
                xtr, xva, _, _ = standardize(train["features"], val["features"])
                _, record = fit(xtr, train["labels"], train["subjects"], **recipe, seed=stable_seed("search", dataset, rid, fold), x_val=xva, y_val=val["labels"], subjects_val=val["subjects"])
                row = {"dataset": dataset, "recipe_id": rid, "fold": fold, **recipe, "validation_BA": record["validation_BA"], "validation_NLL": record["validation_NLL"], "best_epoch": record["best_epoch"]}
                grid_rows.append(row); fold_rows.append(row)
            frame = pd.DataFrame(fold_rows)
            print(f"[decoder-grid] {dataset} {rid+1}/{len(recipes())} {recipe['architecture']} BA={frame.validation_BA.mean():.5f}", flush=True)
        ds_grid = pd.DataFrame([r for r in grid_rows if r["dataset"] == dataset])
        summary = ds_grid.groupby(["recipe_id", "architecture", "lr", "weight_decay", "dropout"], as_index=False).agg(mean_validation_BA=("validation_BA", "mean"), mean_validation_NLL=("validation_NLL", "mean"), minimum_fold_BA=("validation_BA", "min"), median_best_epoch=("best_epoch", "median"))
        selected = summary.sort_values(["mean_validation_BA", "mean_validation_NLL", "minimum_fold_BA", "recipe_id"], ascending=[False, True, False, True]).iloc[0].to_dict()
        selected["final_epochs"] = max(1, int(round(selected["median_best_epoch"])))
        selections.append({"dataset": dataset, **selected})
        recipe = {k: selected[k] for k in ("architecture", "lr", "weight_decay", "dropout")}
        for fold in FOLDS:
            for seed in SEEDS:
                model_fit = load_part(dataset, fold, seed, "model_fit"); validation = load_part(dataset, fold, seed, "validation"); outcome = load_part(dataset, fold, seed, "outcome")
                x_train = np.concatenate([model_fit["features"], validation["features"]]); y_train = np.concatenate([model_fit["labels"], validation["labels"]]); s_train = np.concatenate([model_fit["subjects"], validation["subjects"]])
                x_train, x_outcome, mean, scale = standardize(x_train, outcome["features"])
                model, record = fit(x_train, y_train, s_train, **recipe, seed=stable_seed("final", dataset, fold, seed), fixed_epochs=selected["final_epochs"])
                model.eval(); device = torch.device("cuda")
                with torch.no_grad(): logits = model(torch.from_numpy(x_outcome).float().to(device)).float().cpu().numpy()
                ba = subject_ba(outcome["labels"], logits, outcome["subjects"])
                fold_ba = float(balanced_accuracy_score(outcome["labels"], logits.argmax(1)))
                output_rows.append({"model": "CBraMod", "dataset": dataset, "fold": fold, "seed": seed, **recipe, "epochs": selected["final_epochs"], "held_development_subject_BA": ba, "held_development_trial_BA": fold_ba, "threshold": THRESHOLDS[dataset], "representation_changed": False, "future_session_accessed": False})
                checkpoint = EXP / "runtime" / "decoder" / dataset / f"fold-{fold}" / f"seed-{seed}.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temp = checkpoint.with_suffix(".pt.part"); torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "recipe": recipe, "epochs": selected["final_epochs"], "feature_mean": mean, "feature_scale": scale}, temp); os.replace(temp, checkpoint)
                print(f"[decoder-outcome] {dataset} fold={fold} seed={seed} BA={ba:.5f}", flush=True)
    grid = pd.DataFrame(grid_rows); grid.to_csv(EXP / "runtime" / "CBRAMOD_DECODER_SOURCE_VALIDATION_GRID.csv", index=False)
    pd.DataFrame(selections).to_csv(EXP / "results" / "CBRAMOD_DECODER_SELECTION.csv", index=False)
    outcomes = pd.DataFrame(output_rows); outcomes.to_csv(EXP / "results" / "CBRAMOD_DECODER_PER_FOLD.csv", index=False)
    summary = outcomes.groupby("dataset", as_index=False).agg(BA=("held_development_subject_BA", "mean"), minimum_fold_BA=("held_development_subject_BA", "min"), folds=("fold", "nunique"), seeds=("seed", "nunique"), threshold=("threshold", "first"))
    summary["competent"] = summary.BA >= summary.threshold
    summary["geometry_preserved"] = True
    summary.to_csv(EXP / "results" / "CBRAMOD_COMPETENCE.csv", index=False)
    both = bool(summary.competent.all()); one = bool(summary.competent.any())
    terminal = "CBRAMOD_FROZEN_GEOMETRY_COMPETENT" if both else ("CBRAMOD_FROZEN_GEOMETRY_PARTIAL" if one else "CBRAMOD_COMPETENCE_NOT_RECOVERED")
    result = {"schema": "CBRAMOD_FROZEN_DECODER_RESCUE_V1", "terminal": terminal, "datasets": summary.to_dict("records"), "selection": selections, "representation_changed": False, "scst_geometry_requires_recompute": False, "future_session_accessed": False}
    (EXP / "results" / "CBRAMOD_COMPETENCE.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
