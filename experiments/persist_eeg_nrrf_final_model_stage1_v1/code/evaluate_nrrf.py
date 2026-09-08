"""Phase-A outer-development evaluation, run only after all locked training cells.

This program has no optimizer and never chooses a checkpoint.  New methods use
the locked final epoch-20 EMA checkpoint; raw epoch-20 values are separately
written as diagnostics.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

EXP = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("R2EEG_REPO", EXP.parents[2])).resolve()
RUNTIME = Path(os.environ.get("NRRF_RUNTIME", "/root/rivermind-data/nrrf_final_model_stage1_runtime")).resolve()
SOURCE_RUNTIME = Path(os.environ.get("CARRIER_5FOLD_RUNTIME", "/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")).resolve()
sys.path[:0] = [str(EXP / "code"), str(REPO / "experiments" / "persist_eeg_carrier_dualdataset_screen_v1" / "code"), str(REPO / "experiments" / "persist_eeg_r2eeg_stage1_v1" / "code")]
import run_carrier_screen as carrier  # noqa: E402
import run_stage1 as v1  # noqa: E402
from eegnet_locked import EEGNet  # noqa: E402
from nrrf_model import NRRF  # noqa: E402

OUT = EXP / "outputs"
SPLIT = REPO / "experiments" / "persist_eeg_carrier_5fold_multiseed_stability_v1" / "protocol" / "FIVEFOLD_SPLIT.json"


def selected_path(dataset: str, fold: int, model: str) -> Path:
    return SOURCE_RUNTIME / f"{dataset.lower()}_fold{fold}_seed0_{model.lower()}" / "selected_best.pt"


def make(name: str, channels: int) -> torch.nn.Module:
    return EEGNet(channels) if name == "EEGNet" else carrier.CompactLite(channels, "bn")


def load_source(name: str, channels: int, dataset: str, fold: int, device: torch.device) -> torch.nn.Module:
    path = selected_path(dataset, fold, name)
    model = make(name, channels).to(device)
    bad = model.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=True)
    if bad.missing_keys or bad.unexpected_keys: raise RuntimeError(f"source strict load failed: {path}")
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


def load_new(method: str, dataset: str, fold: int, channels: int, variant: str, device: torch.device) -> NRRF:
    anchor = load_source("EEGNet", channels, dataset, fold, device)
    expressive = load_source("LiteBN", channels, dataset, fold, device)
    path = RUNTIME / "cells" / f"{dataset.lower()}_fold{fold}_seed0_{method.lower().replace('-', '_')}" / f"epoch20_{variant}.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("method") != method or int(payload.get("epochs", -1)) != 20: raise RuntimeError(f"invalid final checkpoint: {path}")
    expressive.load_state_dict(payload["expressive_state"], strict=True)
    return NRRF(anchor, expressive).to(device).eval()


def row_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {"BA": float(balanced_accuracy_score(y, prediction)), "macro_F1": float(f1_score(y, prediction, average="macro", zero_division=0)), "accuracy": float(accuracy_score(y, prediction))}


def subject_logits(model: Any, subject: str, bundle: Any, cache: Any, fusion: bool) -> tuple[np.ndarray, np.ndarray]:
    indices = bundle.indices([subject], (2,)); labels = bundle.labels(indices); outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            x, _ = cache.batch(indices[start:start + 128]); value = model(x)
            outputs.append((value[2] if fusion else value[0]).float().cpu().numpy())
    return labels, np.concatenate(outputs, axis=0)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA required")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    incomplete = []
    for dataset in ("OpenBMI", "WBCIC"):
        for fold in range(5):
            for method in ("JOINT-CE", "NRRF-v1"):
                if not (RUNTIME / "cells" / f"{dataset.lower()}_fold{fold}_seed0_{method.lower().replace('-', '_')}" / "epoch20_ema.pt").is_file(): incomplete.append(f"{dataset}/f{fold}/{method}")
    if incomplete: raise RuntimeError(f"cannot open any outer-development outcome before all 20 cells complete: {incomplete}")
    primary, raw = [], []
    for dataset in ("OpenBMI", "WBCIC"):
        bundle = v1.load_bundle(dataset, list(map(str, split["search_subjects"][dataset])))
        for fold_spec in split["folds"][dataset]:
            fold = int(fold_spec["fold_id"]); mean, std, _ = v1.normalizer(bundle, list(map(str, fold_spec["inner_train_subjects"])))
            cache = carrier.GPUCache(bundle, mean, std, device)
            eeg, lite = load_source("EEGNet", bundle.channels, dataset, fold, device), load_source("LiteBN", bundle.channels, dataset, fold, device)
            models: list[tuple[str, str, Any, bool, bool]] = [
                ("EEGNet", "selected_best", eeg, False, True), ("LiteBN", "selected_best", lite, False, True),
                ("FROZEN-LOGIT50", "selected_best", None, False, True),
                ("JOINT-CE", "EMA_epoch20_PRIMARY", load_new("JOINT-CE", dataset, fold, bundle.channels, "ema", device), True, True),
                ("NRRF-v1", "EMA_epoch20_PRIMARY", load_new("NRRF-v1", dataset, fold, bundle.channels, "ema", device), True, True),
                ("JOINT-CE", "raw_epoch20_diagnostic", load_new("JOINT-CE", dataset, fold, bundle.channels, "raw", device), True, False),
                ("NRRF-v1", "raw_epoch20_diagnostic", load_new("NRRF-v1", dataset, fold, bundle.channels, "raw", device), True, False),
            ]
            for subject in map(str, fold_spec["outer_dev_subjects"]):
                y, eeg_logits = subject_logits(eeg, subject, bundle, cache, fusion=False)
                _, lite_logits = subject_logits(lite, subject, bundle, cache, fusion=False)
                for method, variant, model, fused, is_primary in models:
                    if method == "FROZEN-LOGIT50": logits = .5 * eeg_logits + .5 * lite_logits
                    elif method == "EEGNet": logits = eeg_logits
                    elif method == "LiteBN": logits = lite_logits
                    else: _, logits = subject_logits(model, subject, bundle, cache, fusion=fused)
                    item = {"dataset": dataset, "fold": fold, "seed": 0, "subject_id": subject, "method": method,
                            "checkpoint_variant": variant, "primary": is_primary, "trials": int(len(y)), **row_metrics(y, logits.argmax(axis=1))}
                    (primary if is_primary else raw).append(item)
            del cache, eeg, lite
            torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(primary).to_csv(OUT / "PHASE_A_SUBJECT_RESULTS.csv", index=False)
    pd.DataFrame(raw).to_csv(OUT / "PHASE_A_RAW_EPOCH20_SUBJECT_RESULTS.csv", index=False)
    print("NRRF_PHASE_A_OUTER_EVALUATION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
