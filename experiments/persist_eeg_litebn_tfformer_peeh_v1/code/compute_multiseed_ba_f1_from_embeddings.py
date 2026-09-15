#!/usr/bin/env python3
"""Aggregate frozen classifier BA/Macro-F1 from cached pre-head embeddings.

No neural model is instantiated or executed. Predictions are obtained by one
matrix multiplication through each checkpoint's frozen final linear readout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch


REPO = Path("/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK")
EXP = REPO / "experiments/persist_eeg_litebn_tfformer_peeh_v1"
OUT = EXP / "outputs/peeh_v1"
CACHE = Path("/root/rivermind-data/litebn_tfformer_peeh_v1_runtime/embeddings")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
MODELS = ("LiteBN", "TFFormer")
FOLDS = range(5)
SEEDS = range(3)
CLASSES = {"OpenBMI_MI": 2, "OpenBMI_ERP": 2, "OpenBMI_SSVEP": 4, "WBCIC_MI": 2}
BOOTSTRAP_DRAWS = 20_000


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def subject_sort(values: Sequence[object]) -> list[str]:
    def key(value: object):
        text = str(value)
        tail = text[4:] if text.startswith("sub-") else text
        return (0, int(tail)) if tail.isdigit() else (1, text)
    return sorted(map(str, values), key=key)


def balanced_accuracy(y: np.ndarray, prediction: np.ndarray, classes: int) -> float:
    recalls = [np.mean(prediction[y == label] == label) for label in range(classes) if np.any(y == label)]
    return float(np.mean(recalls))


def macro_f1(y: np.ndarray, prediction: np.ndarray, classes: int) -> float:
    scores = []
    for label in range(classes):
        true_positive = int(np.sum((y == label) & (prediction == label)))
        false_positive = int(np.sum((y != label) & (prediction == label)))
        false_negative = int(np.sum((y == label) & (prediction != label)))
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_readout(model: str, checkpoint: Path, expected_sha256: str) -> tuple[np.ndarray, np.ndarray]:
    if sha256_file(checkpoint) != expected_sha256:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if model == "TFFormer" else payload
    prefix = "base.head" if model == "TFFormer" else "head"
    weight = state[f"{prefix}.weight"].detach().cpu().numpy().astype(np.float64)
    bias = state[f"{prefix}.bias"].detach().cpu().numpy().astype(np.float64)
    if weight.shape[1] != 64 or bias.shape != (weight.shape[0],):
        raise RuntimeError(f"invalid frozen readout shape: {checkpoint} {weight.shape} {bias.shape}")
    return weight, bias


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def scope(task: str) -> str:
    return (
        "true-outer S2 (already accessed)"
        if task == "WBCIC_MI" else "OpenBMI 14-subject internal-heldout diagnostic"
    )


def main() -> None:
    rows = []
    for task in TASKS:
        for model in MODELS:
            for fold in FOLDS:
                for seed in SEEDS:
                    path = CACHE / task / model / f"fold{fold}_seed{seed}.npz"
                    if not path.is_file():
                        raise RuntimeError(f"missing cached embedding; neural inference forbidden: {path}")
                    with np.load(path, allow_pickle=False) as archive:
                        h = np.asarray(archive["eval_h"], np.float64)
                        y = np.asarray(archive["eval_y"], np.int64)
                        subjects = np.asarray(archive["eval_subject"]).astype(str)
                        metadata = json.loads(str(archive["metadata"].item()))
                    if h.ndim != 2 or h.shape[1] != 64 or len(h) != len(y) or len(h) != len(subjects):
                        raise RuntimeError(f"invalid embedding cache schema: {path}")
                    expected = {"task": task, "model": model, "fold": fold, "seed": seed}
                    if any(metadata.get(key) != value for key, value in expected.items()):
                        raise RuntimeError(f"embedding provenance mismatch: {path}")
                    weight, bias = load_readout(model, Path(metadata["checkpoint"]), metadata["checkpoint_sha256"])
                    if weight.shape[0] != CLASSES[task]:
                        raise RuntimeError(f"class/readout mismatch: {path}")
                    prediction = np.argmax(h @ weight.T + bias, axis=1)
                    for subject in subject_sort(np.unique(subjects)):
                        mask = subjects == subject
                        rows.append({
                            "task": task,
                            "model": model,
                            "fold": fold,
                            "seed": seed,
                            "subject_id": subject,
                            "trials": int(mask.sum()),
                            "BA": balanced_accuracy(y[mask], prediction[mask], CLASSES[task]),
                            "Macro_F1": macro_f1(y[mask], prediction[mask], CLASSES[task]),
                            "evaluation_scope": scope(task),
                            "prediction_source": "cached 64-d h + frozen final Linear readout; no neural inference",
                        })

    run_subject = pd.DataFrame(rows)
    subject_seed = run_subject.groupby(["task", "model", "seed", "subject_id"], as_index=False).agg(
        BA=("BA", "mean"), Macro_F1=("Macro_F1", "mean"), repeated_folds=("fold", "size")
    )
    subject = subject_seed.groupby(["task", "model", "subject_id"], as_index=False).agg(
        BA=("BA", "mean"), Macro_F1=("Macro_F1", "mean"), repeated_seeds=("seed", "size")
    )
    seed_summary = subject_seed.groupby(["task", "model", "seed"], as_index=False).agg(
        BA=("BA", "mean"), Macro_F1=("Macro_F1", "mean"), n_biological_subjects=("subject_id", "size")
    )
    summaries = []
    for task in TASKS:
        for model in MODELS:
            group = subject[(subject.task == task) & (subject.model == model)]
            ba_ci = bootstrap(group.BA.to_numpy(float), stable_seed("multiseed-ba", task, model))
            f1_ci = bootstrap(group.Macro_F1.to_numpy(float), stable_seed("multiseed-f1", task, model))
            summaries.append({
                "task": task,
                "model": model,
                "evaluation_scope": scope(task),
                "n_biological_subjects": int(len(group)),
                "BA_mean": float(group.BA.mean()),
                "BA_CI95_L": ba_ci[0],
                "BA_CI95_U": ba_ci[1],
                "Macro_F1_mean": float(group.Macro_F1.mean()),
                "Macro_F1_CI95_L": f1_ci[0],
                "Macro_F1_CI95_U": f1_ci[1],
                "folds_per_seed": 5,
                "seeds": 3,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_unit": "biological subject after within-subject fold/seed averaging",
            })
    summary = pd.DataFrame(summaries)
    OUT.mkdir(parents=True, exist_ok=True)
    run_subject.to_csv(OUT / "MULTISEED_BA_MACRO_F1_RUN_SUBJECT.csv", index=False)
    subject_seed.to_csv(OUT / "MULTISEED_BA_MACRO_F1_SUBJECT_SEED.csv", index=False)
    subject.to_csv(OUT / "MULTISEED_BA_MACRO_F1_SUBJECT.csv", index=False)
    seed_summary.to_csv(OUT / "MULTISEED_BA_MACRO_F1_SEED_SUMMARY.csv", index=False)
    summary.to_csv(OUT / "MULTISEED_BA_MACRO_F1_SUMMARY.csv", index=False)

    lines = [
        "# Frozen LiteBN / TFFormer multi-seed BA and Macro-F1",
        "",
        "No neural inference or retraining was performed. Predictions use cached 64-d h and the frozen final Linear readout.",
        "Five folds and three seeds are averaged within biological subject before 20,000-draw subject bootstrap confidence intervals.",
        "",
        "| Task | Model | BA mean [95% CI] | Macro-F1 mean [95% CI] | N subjects | Scope |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.task} | {row.model} | {100*row.BA_mean:.3f}% [{100*row.BA_CI95_L:.3f}, {100*row.BA_CI95_U:.3f}] | "
            f"{100*row.Macro_F1_mean:.3f}% [{100*row.Macro_F1_CI95_L:.3f}, {100*row.Macro_F1_CI95_U:.3f}] | "
            f"{int(row.n_biological_subjects)} | {row.evaluation_scope} |"
        )
    (OUT / "MULTISEED_BA_MACRO_F1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
