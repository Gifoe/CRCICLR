"""Deterministically replay locked WBCIC large-EEGNet outcome adaptations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from backbones import train_wbcic_large_eegnet as large
from common import ABLATIONS, CACHE, DIAGNOSTICS, LEADERBOARD, PROTOCOL, V6_SEED, logit, sigmoid, stable_seed, write_csv, write_json
from evaluation.metrics import summarize
from protocol.datasets import load_wbcic_fold


def run() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts, replay = [], []
    for fold in range(5):
        payload = torch.load(CACHE / f"WBCIC_LARGE_EEGNET_FOLD_{fold}.pt", map_location="cpu", weights_only=False)
        model = large.LargeEEGNet().to(device)
        model.load_state_dict(payload["model"], strict=True)
        data = load_wbcic_fold(fold)
        for subject in data.outcome_subjects:
            raw = large._load_wbcic_raw(subject)
            frozen = large._predict(model, raw.future_x, device)
            seed = stable_seed(V6_SEED, "WBCIC-large-outcome", fold, subject)
            adapted, _ = large._adapt("wbcic", model, payload["model"], raw, payload["adaptation"], device, seed)
            for method_id, probability, history_used in (
                ("WBCIC_LARGE_EEGNET_FROZEN", frozen, False),
                ("WBCIC_LARGE_EEGNET_TARGET_ADAPTED", adapted, payload["adaptation"]["strategy"] != "frozen"),
            ):
                parts.append(pd.DataFrame({"benchmark": data.benchmark, "method_id": method_id, "trial_uid": raw.future_uid, "subject_id": subject, "outer_fold": fold, "label": raw.future_y, "probability": probability, "prediction": (probability >= 0.5).astype(int), "target_history_labels_used": history_used, "target_future_labels_used_for_fit": False, "exploratory": True, "OUTER_TEST_USED": False}))
            replay.append({"outer_fold": fold, "subject_id": subject, "seed": seed, "adaptation": payload["adaptation"], "OUTER_TEST_USED": False})
        print(f"[WBCIC large deterministic replay] fold={fold} complete", flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    anchor = large._v5()
    aligned = anchor.set_index("trial_uid")
    for method in ("WBCIC_LARGE_EEGNET_FROZEN", "WBCIC_LARGE_EEGNET_TARGET_ADAPTED"):
        part = predictions.loc[predictions.method_id.eq(method)].copy()
        anchor_probability = aligned.loc[part.trial_uid, "probability"].to_numpy(float)
        probability = sigmoid(0.5 * (logit(anchor_probability) + logit(part.probability.to_numpy(float))))
        blend = part.copy()
        blend["method_id"] = "V5_FIXED_BLEND__" + method
        blend["probability"] = probability
        blend["prediction"] = (probability >= 0.5).astype(int)
        blend["target_history_labels_used"] = True
        gate = part.copy()
        gate_probability = np.where(np.abs(anchor_probability - 0.5) <= 0.10, part.probability.to_numpy(float), anchor_probability)
        gate["method_id"] = "V5_UNCERTAINTY_GATE_010__" + method
        gate["probability"] = gate_probability
        gate["prediction"] = (gate_probability >= 0.5).astype(int)
        gate["target_history_labels_used"] = True
        predictions = pd.concat([predictions, blend, gate], ignore_index=True)
    rows, subjects_parts, fold_parts = [], [], []
    anchor_row, anchor_subjects, anchor_folds = summarize(anchor)
    rows.append(anchor_row); subjects_parts.append(anchor_subjects); fold_parts.append(anchor_folds)
    for method in predictions.method_id.unique():
        part = predictions.loc[predictions.method_id.eq(method)].copy()
        row, subjects, folds = summarize(part, reference=anchor)
        rows.append(row); subjects_parts.append(subjects); fold_parts.append(folds)
    table = pd.DataFrame(rows).sort_values("mean_subject_BA", ascending=False)
    write_csv(LEADERBOARD / "WBCIC_LARGE_EEGNET.csv", table)
    write_csv(DIAGNOSTICS / "WBCIC_LARGE_EEGNET_PREDICTIONS.csv", predictions)
    write_csv(DIAGNOSTICS / "WBCIC_LARGE_EEGNET_SUBJECT_RESULTS.csv", pd.concat(subjects_parts, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_LARGE_EEGNET_FOLD_RESULTS.csv", pd.concat(fold_parts, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_LARGE_EEGNET_REPLAY.csv", pd.DataFrame(replay))
    write_csv(ABLATIONS / "WBCIC_BACKBONE_CAPACITY_ABLATION.csv", table)
    write_json(PROTOCOL / "WBCIC_LARGE_EEGNET_DETERMINISTIC_REPLAY_AUDIT.json", {"reason": "dropout RNG was not independently reset in the initial outcome pass", "locked_inputs": "saved refit checkpoints and discovery-selected adaptation configurations", "repair": "per-fold/subject CPU and CUDA RNG reset before adaptation", "outcome_labels_used_for_fit_or_selection": False, "OUTER_TEST_USED": False})
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
