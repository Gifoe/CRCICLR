"""Deterministically replay locked OpenBMI MI outcome adaptations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from backbones import train_openbmi_mi as mi
from common import CACHE, DIAGNOSTICS, LEADERBOARD, PROTOCOL, RESEARCH_LOG, V6_SEED, stable_seed, stage0_root, write_csv, write_json
from evaluation.metrics import summarize
from protocol.datasets import load_openbmi_fold


def run() -> None:
    manifest = pd.read_parquet(stage0_root() / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet")
    manifest = manifest.loc[manifest.paradigm.eq("mi")].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts = []
    replay = []
    for fold in range(5):
        checkpoint = CACHE / f"OPENBMI_MI_SPECIFIC_BACKBONE_FOLD_{fold}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = mi.build(payload["configuration"]).to(device)
        model.load_state_dict(payload["model"], strict=True)
        mean_np, std_np = mi._normalizer(fold)
        mean = torch.as_tensor(mean_np, device=device)
        std = torch.as_tensor(std_np, device=device)
        data = load_openbmi_fold(fold)
        for subject in data.outcome_subjects:
            hx, hy, fx, fy, uid = mi._raw_subject(manifest, subject)
            seed = stable_seed(V6_SEED, "MI-outcome-adapt", fold, subject)
            adapted = mi._adapt(model, hx, hy, payload["adaptation"], device, mean, std, seed)
            probability_parts = []
            adapted.eval()
            with torch.inference_mode():
                for start in range(0, len(fx), 128):
                    xb = mi._normalize(torch.as_tensor(fx[start : start + 128], dtype=torch.float32, device=device), mean, std)
                    probability_parts.append(torch.softmax(adapted(xb), dim=1)[:, 1].cpu().numpy())
            probability = np.concatenate(probability_parts)
            parts.append(pd.DataFrame({"benchmark": data.benchmark, "method_id": "MI_SPECIFIC_BACKBONE_ADAPTED", "trial_uid": uid, "subject_id": subject, "outer_fold": fold, "label": fy, "probability": probability, "prediction": (probability >= 0.5).astype(int), "target_history_labels_used": payload["adaptation"]["strategy"] != "frozen", "target_future_labels_used_for_fit": False, "exploratory": True, "OUTER_TEST_USED": False}))
            replay.append({"outer_fold": fold, "subject_id": subject, "seed": seed, "adaptation": payload["adaptation"], "OUTER_TEST_USED": False})
        print(f"[OpenBMI deterministic replay] fold={fold} complete", flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    baseline = pd.read_csv(DIAGNOSTICS / "OPENBMI_BASELINE_PREDICTIONS.csv")
    reference = baseline.loc[baseline.method_id.eq("B_HISTORY_FUSION_LDA")].copy()
    row, subjects, folds = summarize(predictions, reference=reference)
    table = pd.DataFrame([row])
    write_csv(LEADERBOARD / "OPENBMI_MI_SPECIFIC_BACKBONE.csv", table)
    write_csv(DIAGNOSTICS / "OPENBMI_MI_SPECIFIC_BACKBONE_PREDICTIONS.csv", predictions)
    write_csv(DIAGNOSTICS / "OPENBMI_MI_SPECIFIC_BACKBONE_SUBJECT_RESULTS.csv", subjects)
    write_csv(DIAGNOSTICS / "OPENBMI_MI_SPECIFIC_BACKBONE_FOLD_RESULTS.csv", folds)
    write_csv(DIAGNOSTICS / "OPENBMI_MI_SPECIFIC_BACKBONE_REPLAY.csv", pd.DataFrame(replay))
    write_json(PROTOCOL / "OPENBMI_MI_DETERMINISTIC_REPLAY_AUDIT.json", {"reason": "dropout RNG was not independently reset in the initial outcome pass", "locked_inputs": "saved refit checkpoints and discovery-selected adaptation configurations", "repair": "per-fold/subject CPU and CUDA RNG reset before adaptation", "outcome_labels_used_for_fit_or_selection": False, "OUTER_TEST_USED": False})
    (RESEARCH_LOG / "ITERATION_003_OPENBMI.md").write_text("# Iteration 003 — MI-specific backbone with deterministic outcome replay\n\n```text\n" + table.to_string(index=False) + "\n```\n", encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
