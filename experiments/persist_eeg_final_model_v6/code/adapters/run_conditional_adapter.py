"""Cross-subject episodic conditional-adapter pilot for both benchmarks."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import ABLATIONS, DIAGNOSTICS, LEADERBOARD, PROTOCOL, RESEARCH_LOG, V6_SEED, ensure_directories, stable_seed, write_csv, write_json
from evaluation.metrics import summarize
from history_encoder.conditional_features import fit_context, subject_features
from protocol.datasets import load_fold


VARIANTS = (
    ("A_GENERIC_AFFINE", "affine_only"),
    ("A_GENERIC_BILINEAR", "generic_bilinear"),
    ("A_RANDOM_PROTECTED_CONTROL", "random_protected"),
    ("PERSIST_SA_PROTECTED_BILINEAR", "persist_protected"),
)


class SmallMLP(nn.Module):
    def __init__(self, dimension: int, hidden: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


def _subject_score(labels: np.ndarray, probability: np.ndarray, subjects: np.ndarray) -> tuple[float, float]:
    ba, nll = [], []
    for subject in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject
        ba.append(float(balanced_accuracy_score(labels[mask], probability[mask] >= 0.5)))
        nll.append(float(log_loss(labels[mask], np.clip(probability[mask], 1e-7, 1 - 1e-7), labels=[0, 1])))
    return float(np.mean(ba)), float(np.mean(nll))


def _logistic_candidates(x_train, y_train, x_discovery, y_discovery, discovery_subjects, seed):
    rows = []
    for order, c in enumerate((0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(
            C=c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=4_000,
            random_state=seed + order,
        ).fit(scaler.transform(x_train), y_train)
        probability = model.predict_proba(scaler.transform(x_discovery))[:, 1]
        ba, nll = _subject_score(y_discovery, probability, discovery_subjects)
        rows.append({"family": "logistic", "configuration": {"C": c}, "order": order, "discovery_BA": ba, "discovery_NLL": nll, "model": model, "scaler": scaler})
    return rows


def _mlp_candidates(x_train, y_train, x_discovery, y_discovery, discovery_subjects, seed, device):
    configs = (
        {"hidden": 32, "dropout": 0.1, "weight_decay": 1e-3, "lr": 1e-3},
        {"hidden": 64, "dropout": 0.2, "weight_decay": 1e-3, "lr": 1e-3},
        {"hidden": 128, "dropout": 0.3, "weight_decay": 1e-2, "lr": 5e-4},
    )
    scaler = StandardScaler().fit(x_train)
    x_fit = torch.as_tensor(scaler.transform(x_train), dtype=torch.float32, device=device)
    y_fit = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    x_val = torch.as_tensor(scaler.transform(x_discovery), dtype=torch.float32, device=device)
    rows = []
    for order, config in enumerate(configs):
        torch.manual_seed(seed + order)
        model = SmallMLP(x_train.shape[1], int(config["hidden"]), float(config["dropout"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
        generator = torch.Generator(device="cpu").manual_seed(seed + order)
        best = None
        for epoch in range(100):
            model.train()
            permutation = torch.randperm(len(x_fit), generator=generator, device="cpu").to(device)
            for start in range(0, len(permutation), 256):
                index = permutation[start : start + 256]
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(x_fit[index]), y_fit[index])
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.inference_mode():
                    probability = torch.sigmoid(model(x_val)).cpu().numpy()
                ba, nll = _subject_score(y_discovery, probability, discovery_subjects)
                candidate = (ba, -nll, -(epoch + 1))
                if best is None or candidate > best[0]:
                    best = (candidate, copy.deepcopy(model.state_dict()), epoch + 1, ba, nll)
        assert best is not None
        model.load_state_dict(best[1])
        rows.append(
            {
                "family": "mlp",
                "configuration": {**config, "selected_epoch": best[2]},
                "order": order,
                "discovery_BA": best[3],
                "discovery_NLL": best[4],
                "model": model,
                "scaler": scaler,
            }
        )
    return rows


def run(benchmark: str, device_name: str) -> None:
    ensure_directories()
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    prediction_parts, selection_rows, importance_rows = [], [], []
    for fold in range(5):
        data = load_fold(benchmark, fold)
        context = fit_context(data, stable_seed(V6_SEED, benchmark, fold, "protected"))
        for dimension, (omega, random_omega) in enumerate(zip(context.protected_importance, context.random_importance)):
            importance_rows.append(
                {
                    "benchmark": data.benchmark,
                    "outer_fold": fold,
                    "dimension": dimension,
                    "protected_importance": omega,
                    "random_control_importance": random_omega,
                    "OUTER_TEST_USED": False,
                }
            )
        for method_id, variant in VARIANTS:
            split_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
            for role, subjects in (
                ("train", data.model_fit_subjects),
                ("discovery", data.discovery_subjects),
                ("outcome", data.outcome_subjects),
            ):
                features, labels, subject_ids, trial_ids = [], [], [], []
                for subject in subjects:
                    x, y, uid = subject_features(data, context, subject, variant)
                    features.append(x)
                    labels.append(y)
                    subject_ids.extend([subject] * len(y))
                    trial_ids.extend(uid.tolist())
                split_arrays[role] = (
                    np.concatenate(features),
                    np.concatenate(labels),
                    np.asarray(subject_ids),
                    np.asarray(trial_ids),
                )
            x_train, y_train, _, _ = split_arrays["train"]
            x_discovery, y_discovery, discovery_subjects, _ = split_arrays["discovery"]
            candidates = _logistic_candidates(
                x_train,
                y_train,
                x_discovery,
                y_discovery,
                discovery_subjects,
                stable_seed(V6_SEED, benchmark, fold, variant, "logistic"),
            )
            candidates.extend(
                _mlp_candidates(
                    x_train,
                    y_train,
                    x_discovery,
                    y_discovery,
                    discovery_subjects,
                    stable_seed(V6_SEED, benchmark, fold, variant, "mlp"),
                    device,
                )
            )
            selected = max(candidates, key=lambda row: (row["discovery_BA"], -row["discovery_NLL"], -row["order"], row["family"] == "logistic"))
            x_outcome, y_outcome, outcome_subjects, trial_ids = split_arrays["outcome"]
            if selected["family"] == "logistic":
                probability = selected["model"].predict_proba(selected["scaler"].transform(x_outcome))[:, 1]
            else:
                selected["model"].eval()
                with torch.inference_mode():
                    tensor = torch.as_tensor(selected["scaler"].transform(x_outcome), dtype=torch.float32, device=device)
                    probability = torch.sigmoid(selected["model"](tensor)).cpu().numpy()
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "benchmark": data.benchmark,
                        "method_id": method_id,
                        "trial_uid": trial_ids,
                        "subject_id": outcome_subjects,
                        "outer_fold": fold,
                        "label": y_outcome,
                        "probability": probability,
                        "prediction": (probability >= 0.5).astype(int),
                        "history_sessions": "+".join(map(str, data.history_sessions)),
                        "future_session": data.future_session,
                        "target_history_labels_used": True,
                        "target_future_labels_used_for_fit": False,
                        "exploratory": True,
                        "OUTER_TEST_USED": False,
                    }
                )
            )
            selection_rows.append(
                {
                    "benchmark": data.benchmark,
                    "outer_fold": fold,
                    "method_id": method_id,
                    "family": selected["family"],
                    "configuration": json.dumps(selected["configuration"], sort_keys=True),
                    "discovery_BA": selected["discovery_BA"],
                    "discovery_NLL": selected["discovery_NLL"],
                    "candidate_count": len(candidates),
                    "target_future_labels_used_for_fit": False,
                    "OUTER_TEST_USED": False,
                }
            )
            print(f"[{benchmark}] fold={fold} {method_id} BA={selected['discovery_BA']:.4f} family={selected['family']}", flush=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    if benchmark == "openbmi":
        baseline = pd.read_csv(DIAGNOSTICS / "OPENBMI_BASELINE_PREDICTIONS.csv")
        reference = baseline.loc[baseline.method_id.eq("B_HISTORY_FUSION_LDA")].copy()
    else:
        baseline = pd.read_csv(DIAGNOSTICS / "WBCIC_BASELINE_PREDICTIONS.csv")
        reference = baseline.loc[baseline.method_id.eq("B_HISTORY_FUSION_LDA")].copy()
    rows, subject_parts, fold_parts = [], [], []
    for method in predictions.method_id.unique():
        row, subject_result, fold_result = summarize(predictions.loc[predictions.method_id.eq(method)].copy(), reference=reference)
        rows.append(row)
        subject_parts.append(subject_result)
        fold_parts.append(fold_result)
    table = pd.DataFrame(rows).sort_values("mean_subject_BA", ascending=False)
    write_csv(LEADERBOARD / f"{prefix}_CONDITIONAL_ADAPTER.csv", table)
    write_csv(DIAGNOSTICS / f"{prefix}_CONDITIONAL_ADAPTER_SELECTIONS.csv", pd.DataFrame(selection_rows))
    write_csv(DIAGNOSTICS / f"{prefix}_CONDITIONAL_ADAPTER_PREDICTIONS.csv", predictions)
    write_csv(DIAGNOSTICS / f"{prefix}_CONDITIONAL_ADAPTER_SUBJECT_RESULTS.csv", pd.concat(subject_parts, ignore_index=True))
    write_csv(DIAGNOSTICS / f"{prefix}_CONDITIONAL_ADAPTER_FOLD_RESULTS.csv", pd.concat(fold_parts, ignore_index=True))
    write_csv(DIAGNOSTICS / f"{prefix}_PROTECTED_IMPORTANCE.csv", pd.DataFrame(importance_rows))
    write_csv(ABLATIONS / f"{prefix}_ADAPTER_ABLATION.csv", table)
    write_json(
        PROTOCOL / f"{prefix}_CONDITIONAL_ADAPTER_AUDIT.json",
        {
            "training_unit": "model-fit subject history-to-future episodes",
            "selection_unit": "discovery subject history-to-future episodes",
            "evaluation_unit": "outcome subject future session",
            "history_encoder_supports_variable_K": True,
            "protected_importance_fit_on_model_fit_subjects_only": True,
            "target_future_labels_used_for_fit": False,
            "exploratory": True,
            "OUTER_TEST_USED": False,
        },
    )
    iteration = RESEARCH_LOG / f"ITERATION_001_{prefix}.md"
    iteration.write_text(
        f"# Iteration 001 — {prefix} conditional adapter\n\n"
        "- Failure addressed: target-local last-layer/prototype controls did not substantially beat the population model.\n"
        "- Structural hypothesis: cross-subject history-to-future episodes can learn a bilinear adaptation rule for unseen subjects.\n"
        "- PERSIST intervention: stable model-fit cross-session dimensions are continuously protected; a shuffled equal-capacity control is retained.\n"
        f"- Result:\n\n```text\n{table.to_string(index=False)}\n```\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("openbmi", "wbcic"), required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(args.benchmark, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
