"""Use legal model-fit S3 data to learn a future-session population model.

The frozen WBCIC EEGNet checkpoints were trained on S1/S2 only.  In each
development fold, S3 labels from model-fit subjects are legal meta-training
episodes and should not be discarded.  This script calibrates the population
encoder on those S3 trials, selects capacity on discovery subjects, refits on
all non-outcome subjects, and only then evaluates the outcome subjects.  The
sealed WBCIC outer cohort is never opened.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, log_loss
from torch.utils.data import DataLoader

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from adapters.run_encoder_finetuning import _adapt, _load_wbcic_raw, _trainable, _wbcic_source
from common import ABLATIONS, DIAGNOSTICS, LEADERBOARD, PROTOCOL, RESEARCH_LOG, V6_SEED, stable_seed, v5_output_root, wbcic_source_root, write_csv, write_json
from evaluation.metrics import summarize
from protocol.datasets import load_wbcic_fold


POPULATION_CONFIGS: tuple[dict[str, Any], ...] = (
    {"strategy": "frozen", "lr": 0.0, "epochs": 0},
    {"strategy": "tail", "lr": 3e-4, "epochs": 4},
    {"strategy": "full", "lr": 1e-4, "epochs": 4},
    {"strategy": "full", "lr": 3e-5, "epochs": 12},
)

TARGET_CONFIGS: tuple[dict[str, Any], ...] = (
    {"strategy": "frozen", "lr": 0.0, "epochs": 0, "augment": False},
    {"strategy": "head", "lr": 3e-4, "epochs": 24, "augment": False},
    {"strategy": "tail", "lr": 1e-4, "epochs": 8, "augment": False},
    {"strategy": "full", "lr": 1e-4, "epochs": 8, "augment": False},
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _future_loader(core, subjects: tuple[str, ...], shuffle: bool) -> DataLoader:
    dataset = core.EpochDataset(subjects, (2,))
    return DataLoader(dataset, batch_size=64 if shuffle else 128, shuffle=shuffle, num_workers=0, pin_memory=True)


def _fit_population_s3(
    source_model: nn.Module,
    source_state: dict[str, torch.Tensor],
    core,
    subjects: tuple[str, ...],
    configuration: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, list[float]]:
    model = copy.deepcopy(source_model).to(device)
    model.load_state_dict(source_state, strict=True)
    if configuration["strategy"] == "frozen":
        return model, []
    trainable = _trainable(model, "wbcic", str(configuration["strategy"]))
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=float(configuration["lr"]), weight_decay=1e-4)
    loader = _future_loader(core, subjects, shuffle=True)
    torch.manual_seed(seed)
    losses: list[float] = []
    for _ in range(int(configuration["epochs"])):
        model.train()
        total = 0.0
        count = 0
        for x, y, _, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        losses.append(total / max(count, 1))
    return model, losses


def _predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            logits = model(torch.as_tensor(x[start : start + 128], dtype=torch.float32, device=device))
            parts.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(parts)


def _subject_score(model: nn.Module, raw: dict[str, Any], device: torch.device) -> tuple[float, float]:
    bas, nlls = [], []
    for value in raw.values():
        probability = _predict(model, value.future_x, device)
        bas.append(balanced_accuracy_score(value.future_y, probability >= 0.5))
        nlls.append(log_loss(value.future_y, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1]))
    return float(np.mean(bas)), float(np.mean(nlls))


def _v5_predictions() -> pd.DataFrame:
    path = v5_output_root() / "diagnostics" / "WBCIC_MULTI_SEED_OOF_PREDICTIONS.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame.seed.astype(int).eq(V6_SEED)].copy()
    if len(frame) != 8_195 or frame.trial_uid.duplicated().any() or frame.OUTER_TEST_USED.astype(bool).any():
        raise RuntimeError("Malformed V5 WBCIC anchor")
    frame = frame.rename(columns={"dataset": "benchmark"})
    frame["benchmark"] = "WBCIC_S1S2_to_S3_authorized_development"
    frame["method_id"] = "V5_CS_LGS_ANCHOR"
    frame["target_future_labels_used_for_fit"] = False
    return frame


def run() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core_path = wbcic_source_root() / "experiments" / "persist_eeg_wbcic_actionability_v2" / "code" / "core.py"
    predictions: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    for fold in range(5):
        data = load_wbcic_fold(fold)
        core = _load_module(core_path, f"v6_future_population_core_{fold}")
        source_model, source_state, _, _, checkpoint = _wbcic_source(fold)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected_nonoutcome = set(data.model_fit_subjects) | set(data.discovery_subjects)
        if set(map(str, payload["train_subjects"])) != expected_nonoutcome or list(map(int, payload["train_sessions"])) != [0, 1]:
            raise RuntimeError(f"Fold {fold} source scope mismatch")
        discovery_raw = {subject: _load_wbcic_raw(subject) for subject in data.discovery_subjects}
        outcome_raw = {subject: _load_wbcic_raw(subject) for subject in data.outcome_subjects}
        candidates = []
        for order, configuration in enumerate(POPULATION_CONFIGS):
            model, losses = _fit_population_s3(
                source_model, source_state, core, data.model_fit_subjects, configuration, device,
                stable_seed(V6_SEED, "wbcic-population-S3", fold, order),
            )
            ba, nll = _subject_score(model, discovery_raw, device)
            candidates.append({"order": order, "configuration": configuration, "discovery_BA": ba, "discovery_NLL": nll, "losses": losses, "model": model})
            print(f"[WBCIC S3 population] fold={fold} candidate={order + 1}/{len(POPULATION_CONFIGS)} BA={ba:.4f} {configuration}", flush=True)
        selected = max(candidates, key=lambda row: (row["discovery_BA"], -row["discovery_NLL"], -row["order"]))
        selected_state = {name: value.detach().cpu().clone() for name, value in selected["model"].state_dict().items()}
        adaptation_rows = []
        for order, configuration in enumerate(TARGET_CONFIGS):
            records = []
            for subject, raw in discovery_raw.items():
                probability, _ = _adapt(
                    "wbcic", selected["model"], selected_state, raw, configuration, device,
                    stable_seed(V6_SEED, "wbcic-population-target", fold, order, subject),
                )
                records.append((raw.future_y, probability))
            ba = float(np.mean([balanced_accuracy_score(y, p >= 0.5) for y, p in records]))
            nll = float(np.mean([log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]) for y, p in records]))
            adaptation_rows.append({"order": order, "configuration": configuration, "discovery_BA": ba, "discovery_NLL": nll})
            print(f"[WBCIC S3 target] fold={fold} candidate={order + 1}/{len(TARGET_CONFIGS)} BA={ba:.4f} {configuration}", flush=True)
        selected_adaptation = max(adaptation_rows, key=lambda row: (row["discovery_BA"], -row["discovery_NLL"], -row["order"]))
        refit_model, refit_losses = _fit_population_s3(
            source_model, source_state, core, tuple(sorted(expected_nonoutcome)), selected["configuration"], device,
            stable_seed(V6_SEED, "wbcic-population-S3-refit", fold),
        )
        refit_state = {name: value.detach().cpu().clone() for name, value in refit_model.state_dict().items()}
        for subject, raw in outcome_raw.items():
            population_probability = _predict(refit_model, raw.future_x, device)
            adapted_probability, _ = _adapt(
                "wbcic", refit_model, refit_state, raw, selected_adaptation["configuration"], device,
                stable_seed(V6_SEED, "wbcic-population-target-outcome", fold, subject),
            )
            for method_id, probability, history_used in (
                ("B_FUTURE_SESSION_POPULATION", population_probability, False),
                ("A_FUTURE_SESSION_TARGET_ADAPTED", adapted_probability, selected_adaptation["configuration"]["strategy"] != "frozen"),
            ):
                predictions.append(
                    pd.DataFrame(
                        {
                            "benchmark": data.benchmark,
                            "method_id": method_id,
                            "trial_uid": raw.future_uid,
                            "subject_id": subject,
                            "outer_fold": fold,
                            "label": raw.future_y,
                            "probability": probability,
                            "prediction": (probability >= 0.5).astype(int),
                            "target_history_labels_used": history_used,
                            "target_future_labels_used_for_fit": False,
                            "exploratory": True,
                            "OUTER_TEST_USED": False,
                        }
                    )
                )
        selection_rows.extend(
            {
                "outer_fold": fold,
                "stage": "population_candidate",
                "candidate_order": row["order"],
                "configuration": json.dumps(row["configuration"], sort_keys=True),
                "discovery_BA": row["discovery_BA"],
                "discovery_NLL": row["discovery_NLL"],
                "selected": row is selected,
                "training_losses": json.dumps(row["losses"]),
                "target_future_labels_used_for_fit": False,
                "OUTER_TEST_USED": False,
            }
            for row in candidates
        )
        selection_rows.extend(
            {
                "outer_fold": fold,
                "stage": "target_adaptation_candidate",
                "candidate_order": row["order"],
                "configuration": json.dumps(row["configuration"], sort_keys=True),
                "discovery_BA": row["discovery_BA"],
                "discovery_NLL": row["discovery_NLL"],
                "selected": row is selected_adaptation,
                "training_losses": None,
                "target_future_labels_used_for_fit": False,
                "OUTER_TEST_USED": False,
            }
            for row in adaptation_rows
        )
        print(f"[WBCIC S3] fold={fold} selected_population={selected['configuration']} selected_target={selected_adaptation['configuration']} refit_last_loss={refit_losses[-1] if refit_losses else None}", flush=True)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    v5 = _v5_predictions()
    # Fixed, untuned anchor blends are legal and expose complementarity without
    # using any outcome label to select a weight.
    aligned_v5 = v5.set_index("trial_uid")
    for method in ("B_FUTURE_SESSION_POPULATION", "A_FUTURE_SESSION_TARGET_ADAPTED"):
        part = prediction_frame.loc[prediction_frame.method_id.eq(method)].copy()
        anchor = aligned_v5.loc[part.trial_uid, "probability"].to_numpy(float)
        candidate = np.clip(part.probability.to_numpy(float), 1e-7, 1 - 1e-7)
        anchor = np.clip(anchor, 1e-7, 1 - 1e-7)
        probability = 1.0 / (1.0 + np.exp(-0.5 * ((np.log(anchor) - np.log1p(-anchor)) + (np.log(candidate) - np.log1p(-candidate)))))
        part["method_id"] = f"V5_FIXED_LOGIT_BLEND__{method}"
        part["probability"] = probability
        part["prediction"] = (probability >= 0.5).astype(int)
        part["target_history_labels_used"] = True
        prediction_frame = pd.concat([prediction_frame, part], ignore_index=True)
    reference = v5.copy()
    rows, subject_parts, fold_parts = [], [], []
    v5_row, v5_subjects, v5_folds = summarize(reference)
    rows.append(v5_row); subject_parts.append(v5_subjects); fold_parts.append(v5_folds)
    for method in prediction_frame.method_id.unique():
        part = prediction_frame.loc[prediction_frame.method_id.eq(method)].copy()
        row, subjects, folds = summarize(part, reference=reference)
        rows.append(row); subject_parts.append(subjects); fold_parts.append(folds)
    table = pd.DataFrame(rows).sort_values("mean_subject_BA", ascending=False)
    write_csv(LEADERBOARD / "WBCIC_FUTURE_SESSION_POPULATION.csv", table)
    write_csv(DIAGNOSTICS / "WBCIC_FUTURE_SESSION_POPULATION_SELECTIONS.csv", pd.DataFrame(selection_rows))
    write_csv(DIAGNOSTICS / "WBCIC_FUTURE_SESSION_POPULATION_PREDICTIONS.csv", prediction_frame)
    write_csv(DIAGNOSTICS / "WBCIC_FUTURE_SESSION_POPULATION_SUBJECT_RESULTS.csv", pd.concat(subject_parts, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_FUTURE_SESSION_POPULATION_FOLD_RESULTS.csv", pd.concat(fold_parts, ignore_index=True))
    write_csv(ABLATIONS / "WBCIC_FUTURE_SESSION_TRAINING_ABLATION.csv", table)
    write_json(
        PROTOCOL / "WBCIC_FUTURE_SESSION_POPULATION_AUDIT.json",
        {
            "source": "fold-specific EEGNet trained on all non-outcome subjects S1/S2",
            "model_fit_episode": "S3 labels from model-fit subjects only",
            "selection": "discovery-subject S3 only",
            "refit": "fixed configuration on all non-outcome S3",
            "outcome_target_adaptation": "labeled S1/S2 only",
            "outcome_S3_labels_used_for_fit_or_selection": False,
            "V5_blend_weight": 0.5,
            "V5_blend_weight_tuned": False,
            "WBCIC_outer_split_file_opened": False,
            "OUTER_TEST_USED": False,
        },
    )
    (RESEARCH_LOG / "ITERATION_004_WBCIC.md").write_text(
        "# Iteration 004 — legal future-session population training\n\n"
        "The previous source EEGNet discarded legal S3 labels from model-fit subjects.  This iteration uses those subjects as history-to-future meta-training episodes, then applies the same target-history adaptation ladder.\n\n"
        "```text\n" + table.to_string(index=False) + "\n```\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
