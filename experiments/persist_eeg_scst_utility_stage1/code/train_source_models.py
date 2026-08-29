"""Source-only training and representation export for official Stage-1 models."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import stage1_common as c


MODELS = ("ATCNet-Official", "EEGNeX")
RECIPES = (
    {"id": "official_a", "lr": 1e-3, "weight_decay": 1e-3, "max_epochs": 30, "batch_size": 192},
    {"id": "official_b", "lr": 3e-4, "weight_decay": 1e-2, "max_epochs": 36, "batch_size": 192},
)


def grid_checkpoint(model: str, dataset: str, recipe: str, fold: int, seed: int) -> Path:
    return c.RUNTIME / "source_grid" / model / dataset / recipe / f"fold-{fold}" / f"seed-{seed}.pt"


def train_one(model_name: str, dataset: str, recipe: dict[str, object], fold: int, seed: int, device: torch.device) -> dict[str, object]:
    path = grid_checkpoint(model_name, dataset, str(recipe["id"]), fold, seed)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("complete") is True:
            return payload["record"]
    raw, metadata, _ = c.load_data(dataset)
    role = c.roles(dataset, fold)
    source_sessions = c.SOURCE_SESSIONS[dataset]
    train_idx = c.row_indices(metadata, role["model_fit"], source_sessions)
    val_idx = c.row_indices(metadata, role["validation"], source_sessions)
    labels = metadata.label.to_numpy(np.int64)
    c.set_seed(c.stable_seed("stage1-source", model_name, dataset, recipe["id"], fold, seed))
    net = c.build_model(model_name, raw.shape[1], raw.shape[2]).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(c.stable_seed("stage1-order", model_name, dataset, recipe["id"], fold, seed))
    best_key = (-math.inf, -math.inf)
    best_state = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    max_epochs = int(recipe["max_epochs"])
    batch_size = int(recipe["batch_size"])
    for epoch in range(max_epochs):
        net.train()
        order = rng.permutation(train_idx)
        losses = []
        factor = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * epoch / max_epochs))
        for group in optimizer.param_groups:
            group["lr"] = float(recipe["lr"]) * factor
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            x = torch.from_numpy(c.normalize_raw(raw[idx])).to(device)
            y = torch.as_tensor(labels[idx], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = net(x)
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = c.infer_model(model_name, net, raw, metadata, val_idx, None, None, device, batch_size)
        score = c.metrics(val["labels"], val["logits"], val["subjects"])
        key = (score["BA"], -score["NLL"])
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_BA": score["BA"], "validation_NLL": score["NLL"]})
        if key > best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        print(f"[source] {model_name} {dataset} {recipe['id']} f={fold} s={seed} e={epoch+1} valBA={score['BA']:.5f} best={best_key[0]:.5f}", flush=True)
        if epoch + 1 >= 12 and stale >= 7:
            break
    record = {
        "model": model_name, "dataset": dataset, "recipe": recipe["id"], "fold": fold, "seed": seed,
        "validation_BA": float(best_key[0]), "validation_NLL": float(-best_key[1]), "best_epoch": best_epoch,
        "parameters": int(sum(p.numel() for p in net.parameters())), "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)), "future_session_used": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.part")
    torch.save({"complete": True, "state_dict": best_state, "recipe": dict(recipe), "record": record, "history": history}, temporary)
    os.replace(temporary, path)
    del net
    torch.cuda.empty_cache()
    return record


def load_one(model_name: str, dataset: str, path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw, _, _ = c.load_data(dataset)
    net = c.build_model(model_name, raw.shape[1], raw.shape[2])
    net.load_state_dict(payload["state_dict"])
    return net.to(device).eval(), payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=c.DATASETS, default=list(c.DATASETS))
    args = parser.parse_args()
    c.ensure_dirs()
    run_tag = "__".join(value.lower().replace("-", "_") for value in args.models)
    device = torch.device("cuda")
    all_runs: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    competence: list[dict[str, object]] = []
    for model_name in args.models:
        for dataset in args.datasets:
            seed0: list[dict[str, object]] = []
            for recipe in RECIPES:
                for fold in c.FOLDS:
                    row = train_one(model_name, dataset, recipe, fold, 0, device)
                    seed0.append(row)
                    all_runs.append(row)
            grid = pd.DataFrame(seed0).groupby("recipe", as_index=False).agg(
                mean_validation_BA=("validation_BA", "mean"),
                mean_validation_NLL=("validation_NLL", "mean"),
                minimum_fold_BA=("validation_BA", "min"),
            )
            selected = grid.sort_values(
                ["mean_validation_BA", "mean_validation_NLL", "minimum_fold_BA", "recipe"],
                ascending=[False, True, False, True],
            ).iloc[0]
            chosen = next(value for value in RECIPES if value["id"] == selected.recipe)
            selections.append({"model": model_name, "dataset": dataset, **selected.to_dict(), "selected_before_source_outcome": True, "future_session_used": False})
            for fold in c.FOLDS:
                for seed in c.SEEDS:
                    row = train_one(model_name, dataset, chosen, fold, seed, device)
                    if not any(r["model"] == model_name and r["dataset"] == dataset and r["recipe"] == chosen["id"] and r["fold"] == fold and r["seed"] == seed for r in all_runs):
                        all_runs.append(row)
                    checkpoint = grid_checkpoint(model_name, dataset, str(chosen["id"]), fold, seed)
                    net, _ = load_one(model_name, dataset, checkpoint, device)
                    raw, metadata, _ = c.load_data(dataset)
                    role = c.roles(dataset, fold)
                    partitions = {}
                    for part in ("model_fit", "validation", "outcome"):
                        idx = c.row_indices(metadata, role[part], c.SOURCE_SESSIONS[dataset])
                        rep = c.infer_model(model_name, net, raw, metadata, idx, None, None, device, int(chosen["batch_size"]))
                        c.save_rep(c.rep_path(model_name, dataset, fold, seed, part), rep)
                        partitions[part] = rep
                    score = c.metrics(partitions["outcome"]["labels"], partitions["outcome"]["logits"], partitions["outcome"]["subjects"])
                    competence.append({
                        "model": model_name, "dataset": dataset, "fold": fold, "seed": seed, "recipe": chosen["id"],
                        "BA": score["BA"], "macro_F1": score["macro_F1"], "NLL": score["NLL"],
                        "threshold": c.THRESHOLDS[dataset], "competent": bool(score["BA"] >= c.THRESHOLDS[dataset]),
                        "parameters": row["parameters"], "future_session_used": False,
                    })
                    print(f"[source-outcome] {model_name} {dataset} f={fold} s={seed} BA={score['BA']:.5f}", flush=True)
                    del net
                    torch.cuda.empty_cache()
            c.write_csv(c.RUNTIME / f"SOURCE_TRAINING_RUNS_{run_tag}.csv", pd.DataFrame(all_runs))
            c.write_csv(c.RESULTS / f"SOURCE_SELECTION_{run_tag}.csv", pd.DataFrame(selections))
            c.write_csv(c.RESULTS / f"SOURCE_COMPETENCE_PER_FOLD_{run_tag}.csv", pd.DataFrame(competence))
    frame = pd.DataFrame(competence)
    summary = frame.groupby(["model", "dataset"], as_index=False).agg(
        BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), NLL=("NLL", "mean"),
        threshold=("threshold", "first"), folds=("fold", "nunique"), seeds=("seed", "nunique"), parameters=("parameters", "first"),
    )
    summary["competent"] = summary.BA >= summary.threshold
    c.write_csv(c.RESULTS / f"SOURCE_COMPETENCE_{run_tag}.csv", summary)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
