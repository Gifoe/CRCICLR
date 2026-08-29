"""Prospective matched WBCIC session-2 utility experiment.

The script refuses to run without the committed Stage-1 training lock. It uses
the frozen source anchor and a fixed Option-A transport bank; no future outcome
is used for selection or early stopping.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

import audit_stage1 as audit
import stage1_common as c


METHODS = ("ERM", "Mixup", "RandomTransport", "SCST-NoConsistency", "Full-SCST")
EPOCHS = 15
BATCH_SIZE = 192
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
LAMBDA_T = 0.5
LAMBDA_C = 0.1


def old_selection(model: str, dataset: str) -> str:
    frame = pd.read_csv(c.REPO / "experiments" / "persist_eeg_scst_competence_generality_v1" / "results" / "SPECIALIST_SELECTION.csv")
    return str(frame[(frame.model == model) & (frame.dataset == dataset)].iloc[0].config)


def source_selection(model: str, dataset: str) -> str:
    tag = model.lower().replace("-", "_")
    frame = pd.read_csv(c.RESULTS / f"SOURCE_SELECTION_{tag}.csv")
    return str(frame[(frame.model == model) & (frame.dataset == dataset)].iloc[0].recipe)


def anchor_path(model: str, fold: int, seed: int) -> Path:
    if model == "ATCNet-CleanRoom":
        recipe = old_selection("ATCNet", "WBCIC")
        return c.REPO / "experiments" / "persist_eeg_scst_competence_generality_v1" / "runtime" / "specialist_checkpoints" / "ATCNet" / "WBCIC" / recipe / f"fold-{fold}" / f"seed-{seed}.pt"
    recipe = source_selection(model, "WBCIC")
    return c.RUNTIME / "source_grid" / model / "WBCIC" / recipe / f"fold-{fold}" / f"seed-{seed}.pt"


def load_anchor(model: str, fold: int, seed: int, device: torch.device):
    path = anchor_path(model, fold, seed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw, _, _ = c.load_data("WBCIC")
    net = c.build_model(model, raw.shape[1], raw.shape[2])
    net.load_state_dict(payload["state_dict"])
    return net.to(device), path


def source_rep(model: str, fold: int, seed: int, role: str) -> dict[str, np.ndarray]:
    return audit.load_rep(model, "WBCIC", fold, seed, role)


def combine_source(model: str, fold: int, seed: int) -> dict[str, np.ndarray]:
    values = [source_rep(model, fold, seed, role) for role in ("model_fit", "validation")]
    keys = ("indices", "features", "logits", "labels", "subjects", "sessions")
    result = {key: np.concatenate([value[key] for value in values]) for key in keys}
    if len(np.unique(result["indices"])) != len(result["indices"]):
        raise RuntimeError("source training rows overlap")
    return result


def centroids(rep: dict[str, np.ndarray]) -> dict[tuple[str, int], np.ndarray]:
    result = {}
    for subject in c.subject_sort(np.unique(rep["subjects"].astype(str))):
        for label in sorted(np.unique(rep["labels"]).astype(int).tolist()):
            mask = (rep["subjects"].astype(str) == subject) & (rep["labels"].astype(int) == label) & (rep["sessions"].astype(int) == 0)
            if not mask.any():
                raise RuntimeError(f"missing source bank cell {subject=} {label=}")
            result[(subject, label)] = rep["features"][mask].mean(0).astype(np.float64)
    return result


def frozen_transport(rep: dict[str, np.ndarray], model: str, fold: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    features = rep["features"].astype(np.float64)
    bank_mask = rep["sessions"].astype(int) == 0
    center = features[bank_mask].mean(0)
    scale = features[bank_mask].std(0)
    scale[scale < 1e-6] = 1.0
    normalized = (features - center) / scale
    normalized_rep = {**rep, "features": normalized}
    cs = centroids(normalized_rep)
    subjects = c.subject_sort(np.unique(rep["subjects"].astype(str)))
    labels = sorted(np.unique(rep["labels"]).astype(int).tolist())
    population = {label: np.mean([cs[(subject, label)] for subject in subjects], 0) for label in labels}
    residual = {(subject, label): cs[(subject, label)] - population[label] for subject in subjects for label in labels}
    transported_delta = np.empty_like(features, dtype=np.float32)
    random_delta = np.empty_like(features, dtype=np.float32)
    selected_alpha = np.empty(len(features), dtype=np.float32)
    rng = np.random.default_rng(c.stable_seed("stage1-random-bank", model, fold, seed))
    for label in labels:
        support = np.stack([cs[(subject, label)] for subject in subjects])
        radius = audit.support_radius(support)
        label_rows = np.flatnonzero(rep["labels"].astype(int) == label)
        for row in label_rows:
            subject = str(rep["subjects"][row])
            candidates = [value for value in subjects if value != subject]
            target = candidates[c.stable_seed("stage1-target", model, fold, seed, int(rep["indices"][row])) % len(candidates)]
            delta_z = residual[(target, label)] - residual[(subject, label)]
            query_z = normalized[row]
            alpha = float(audit.solve_alpha(query_z[None], delta_z[None], support, radius)[0])
            noise_z = rng.normal(size=len(delta_z))
            noise_z *= np.linalg.norm(delta_z) / max(np.linalg.norm(noise_z), audit.EPS)
            transported_delta[row] = (alpha * delta_z * scale).astype(np.float32)
            random_delta[row] = (alpha * noise_z * scale).astype(np.float32)
            selected_alpha[row] = alpha
    return transported_delta, random_delta, {
        "alpha_mean": float(selected_alpha.mean()), "alpha_nonzero": float(np.mean(selected_alpha > 0)),
        "transport_norm_mean": float(np.linalg.norm(transported_delta / scale, axis=1).mean()),
        "random_norm_mean": float(np.linalg.norm(random_delta / scale, axis=1).mean()),
    }


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log_softmax(logits.float(), -1)).sum(-1).mean()


def unit_path(model: str, method: str, fold: int, seed: int) -> Path:
    return c.RUNTIME / "utility_units" / model / method / f"fold-{fold}" / f"seed-{seed}.pt"


def train_method(model: str, method: str, fold: int, seed: int, source: dict[str, np.ndarray], delta: np.ndarray, random_delta: np.ndarray, device: torch.device):
    path = unit_path(model, method, fold, seed)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("complete") is True:
            net, _ = load_anchor(model, fold, seed, device)
            net.load_state_dict(payload["state_dict"])
            return net.eval(), payload["history"]
    net, anchor = load_anchor(model, fold, seed, device)
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    raw, _, _ = c.load_data("WBCIC")
    indices = source["indices"].astype(np.int64)
    labels = source["labels"].astype(np.int64)
    rng = np.random.default_rng(c.stable_seed("stage1-utility-order", model, fold, seed))
    method_rng = np.random.default_rng(c.stable_seed("stage1-utility-method", model, fold, seed, method))
    history = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(indices))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            pos = order[start : start + BATCH_SIZE]
            x = torch.from_numpy(c.normalize_raw(raw[indices[pos]])).to(device)
            y = torch.as_tensor(labels[pos], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = c.model_features(model, net, x)
                clean_logits = c.feature_logits(model, net, h)
                clean_loss = F.cross_entropy(clean_logits.float(), y)
                loss = clean_loss
                if method == "Mixup":
                    permutation = torch.as_tensor(method_rng.permutation(len(pos)), dtype=torch.long, device=device)
                    weight = float(method_rng.beta(0.4, 0.4))
                    mixed_h = weight * h + (1.0 - weight) * h[permutation]
                    target = weight * F.one_hot(y, 2).float() + (1.0 - weight) * F.one_hot(y[permutation], 2).float()
                    loss = clean_loss + LAMBDA_T * soft_ce(c.feature_logits(model, net, mixed_h), target)
                elif method in ("RandomTransport", "SCST-NoConsistency", "Full-SCST"):
                    bank = random_delta if method == "RandomTransport" else delta
                    moved_h = h + torch.from_numpy(bank[pos]).to(device)
                    moved_logits = c.feature_logits(model, net, moved_h)
                    loss = clean_loss + LAMBDA_T * F.cross_entropy(moved_logits.float(), y)
                    if method == "Full-SCST":
                        loss = loss + LAMBDA_C * c.symmetric_kl(clean_logits, moved_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses))})
        print(f"[utility-train] {model} {method} f={fold} s={seed} e={epoch+1} loss={np.mean(losses):.5f}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.part")
    torch.save({"complete": True, "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "history": history, "anchor": str(anchor)}, temporary)
    os.replace(temporary, path)
    return net.eval(), history


def evaluate_future(model: str, net, fold: int, device: torch.device) -> list[dict[str, object]]:
    raw, metadata, _ = c.load_data("WBCIC")
    role = c.roles("WBCIC", fold)
    idx = c.row_indices(metadata, role["outcome"], (2,))
    rep = c.infer_model(model, net, raw, metadata, idx, None, None, device, BATCH_SIZE)
    pred = rep["logits"].argmax(1)
    stable = rep["logits"].astype(np.float64) - rep["logits"].max(1, keepdims=True)
    probability = np.exp(stable); probability /= probability.sum(1, keepdims=True)
    rows = []
    for subject in c.subject_sort(np.unique(rep["subjects"].astype(str))):
        mask = rep["subjects"].astype(str) == subject
        rows.append({
            "model": model, "fold": fold, "subject_id": subject,
            "BA": float(balanced_accuracy_score(rep["labels"][mask], pred[mask])),
            "macro_F1": float(f1_score(rep["labels"][mask], pred[mask], average="macro", zero_division=0)),
            "CE": float(log_loss(rep["labels"][mask], probability[mask], labels=[0, 1])),
            "trials": int(mask.sum()), "future_session": 2,
        })
    return rows


def aggregate(model: str) -> None:
    unit_files = sorted((c.RUNTIME / "utility_metrics" / model).rglob("*.csv"))
    frame = pd.concat([pd.read_csv(path) for path in unit_files], ignore_index=True)
    c.write_csv(c.RESULTS / f"SCST_PER_SUBJECT_{model}.csv", frame)
    per_fold = frame.groupby(["model", "method", "fold", "seed"], as_index=False).agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), CE=("CE", "mean"), subjects=("subject_id", "nunique"))
    c.write_csv(c.RESULTS / f"SCST_PER_FOLD_{model}.csv", per_fold)
    piv = frame.pivot_table(index=["fold", "seed", "subject_id"], columns="method", values="BA").reset_index()
    subject = piv.groupby("subject_id", as_index=False).mean(numeric_only=True)
    rng = np.random.default_rng(c.stable_seed("stage1-subject-bootstrap", model))
    summary = []
    for method in METHODS:
        values = subject[method].to_numpy(np.float64)
        draws = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(1)
        summary.append({"model": model, "method": method, "BA": float(values.mean()), "median_subject_BA": float(np.median(values)), "CI95_L": float(np.quantile(draws, .025)), "CI95_U": float(np.quantile(draws, .975))})
    deltas = []
    for method in METHODS[1:]:
        values = (subject[method] - subject["ERM"]).to_numpy(np.float64)
        draws = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(1)
        fold_delta = piv.groupby("fold").apply(lambda value: float((value[method] - value["ERM"]).mean()), include_groups=False)
        deltas.append({
            "model": model, "comparison": f"{method}-ERM", "delta_BA": float(values.mean()),
            "CI95_L": float(np.quantile(draws, .025)), "CI95_U": float(np.quantile(draws, .975)),
            "positive_folds": int((fold_delta > 0).sum()), "folds": int(len(fold_delta)),
        })
    full_random = float((subject["Full-SCST"] - subject["RandomTransport"]).mean())
    full = next(value for value in deltas if value["comparison"] == "Full-SCST-ERM")
    success = bool(full["delta_BA"] > 0 and full["CI95_L"] > 0 and full["positive_folds"] >= 3 and full_random > 0)
    c.write_csv(c.RESULTS / f"SCST_SUMMARY_{model}.csv", pd.DataFrame(summary))
    c.write_csv(c.RESULTS / f"CONTROL_COMPARISON_{model}.csv", pd.DataFrame(deltas))
    c.write_json(c.RESULTS / f"STATISTICS_{model}.json", {"model": model, "bootstrap_draws": 10000, "comparisons": deltas, "full_scst_minus_random": full_random, "SCST_POSITIVE": success})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("ATCNet-CleanRoom", "ATCNet-Official", "EEGNeX"))
    args = parser.parse_args()
    lock_path = c.PROTOCOL / "SCST_STAGE1_TRAINING_LOCK.json"
    if not lock_path.is_file():
        raise RuntimeError("SCST_STAGE1_TRAINING_LOCK_MISSING")
    lock = c.read_json(lock_path)
    if lock.get("future_utility_accessed_before_lock") is not False or args.model not in lock.get("eligible_models", []):
        raise RuntimeError("MODEL_NOT_PROSPECTIVELY_AUTHORIZED")
    device = torch.device("cuda")
    for fold in c.FOLDS:
        for seed in c.SEEDS:
            source = combine_source(args.model, fold, seed)
            delta, random_delta, bank_stats = frozen_transport(source, args.model, fold, seed)
            for method in METHODS:
                metric_path = c.RUNTIME / "utility_metrics" / args.model / method / f"fold-{fold}" / f"seed-{seed}.csv"
                if metric_path.is_file():
                    continue
                net, _ = train_method(args.model, method, fold, seed, source, delta, random_delta, device)
                rows = evaluate_future(args.model, net, fold, device)
                for row in rows:
                    row.update({"seed": seed, "method": method, **bank_stats})
                c.write_csv(metric_path, pd.DataFrame(rows))
                print(f"[future-eval] {args.model} {method} f={fold} s={seed} subjects={len(rows)}", flush=True)
                del net
                torch.cuda.empty_cache()
    aggregate(args.model)


if __name__ == "__main__":
    main()
