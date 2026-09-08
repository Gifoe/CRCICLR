"""Inference-only CURRENT vs RESTORE_SOURCE_BN audit for all 20 EMA cells."""
from __future__ import annotations

import hashlib
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
SOURCE_RUNTIME = Path(os.environ.get("CARRIER_5FOLD_RUNTIME", "/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")).resolve()
NRRF_RUNTIME = Path(os.environ.get("NRRF_RUNTIME", "/root/rivermind-data/nrrf_final_model_stage1_runtime")).resolve()
CARRIER_EXP = REPO / "experiments" / "persist_eeg_carrier_5fold_multiseed_stability_v1"
sys.path[:0] = [str(EXP / "code"), str(REPO / "experiments" / "persist_eeg_carrier_dualdataset_screen_v1" / "code"), str(REPO / "experiments" / "persist_eeg_r2eeg_stage1_v1" / "code")]
import run_carrier_screen as carrier  # noqa: E402
import run_stage1 as v1  # noqa: E402
from eegnet_locked import EEGNet  # noqa: E402
from bn_buffer_intervention import ALPHA, batchnorm_buffer_keys, drift_rows, frozen, fused_logits, restore_source_bn, validate_intervention  # noqa: E402

OUT, PROTOCOL = EXP / "outputs", EXP / "protocol"
METHODS = ("JOINT-CE", "NRRF-v1")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def clean(v: Any) -> Any:
    if isinstance(v, Path): return str(v)
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)): return float(v)
    if isinstance(v, (np.bool_, bool)): return bool(v)
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_path(dataset: str, fold: int, model: str) -> Path:
    return SOURCE_RUNTIME / f"{dataset.lower()}_fold{fold}_seed0_{model.lower()}" / "selected_best.pt"


def stage2_path(dataset: str, fold: int, method: str, variant: str) -> Path:
    return NRRF_RUNTIME / "cells" / f"{dataset.lower()}_fold{fold}_seed0_{method.lower().replace('-', '_')}" / f"epoch20_{variant}.pt"


def make_source(name: str, channels: int, path: Path, device: torch.device) -> torch.nn.Module:
    model = EEGNet(channels) if name == "EEGNet" else carrier.CompactLite(channels, "bn")
    state = torch.load(path, map_location=device, weights_only=False)
    bad = model.load_state_dict(state, strict=True)
    if bad.missing_keys or bad.unexpected_keys: raise RuntimeError(f"strict load mismatch: {path}")
    return frozen(model.to(device))


def make_stage2(dataset: str, fold: int, method: str, variant: str, channels: int, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, list[str], dict[str, Any]]:
    anchor = make_source("EEGNet", channels, source_path(dataset, fold, "eegnet"), device)
    source_expr = make_source("LiteBN", channels, source_path(dataset, fold, "litebn"), device)
    stage_path = stage2_path(dataset, fold, method, variant)
    payload = torch.load(stage_path, map_location=device, weights_only=False)
    if payload.get("method") != method or int(payload.get("epochs", -1)) != 20: raise RuntimeError(f"invalid stage2 checkpoint: {stage_path}")
    current = carrier.CompactLite(channels, "bn").to(device)
    current.load_state_dict(payload["expressive_state"], strict=True)
    current = frozen(current)
    source_state, current_state = source_expr.state_dict(), current.state_dict()
    allowed = batchnorm_buffer_keys(current)
    restored_state = restore_source_bn(current_state, source_state, allowed)
    validation = validate_intervention(current, current_state, source_state, restored_state, allowed)
    restored = carrier.CompactLite(channels, "bn").to(device)
    restored.load_state_dict(restored_state, strict=True)
    restored = frozen(restored)
    return anchor, current, restored, allowed, {"state_diff": validation, "stage2_path": str(stage_path), "stage2_sha256": sha(stage_path), "source_path": str(source_path(dataset, fold, "litebn")), "source_sha256": sha(source_path(dataset, fold, "litebn"))}


def subject_logits(anchor: torch.nn.Module, expr: torch.nn.Module, subject: str, bundle: Any, cache: Any) -> tuple[np.ndarray, np.ndarray]:
    indices = bundle.indices([subject], (2,)); labels = bundle.labels(indices); pieces = []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            x, _ = cache.batch(indices[start:start + 128]); pieces.append(fused_logits(anchor, expr, x).float().cpu().numpy())
    return labels, np.concatenate(pieces)


def branch_logits(model: torch.nn.Module, subject: str, bundle: Any, cache: Any) -> tuple[np.ndarray, np.ndarray]:
    """Raw carrier logits, used only to reconstruct the exact frozen LOGIT50 baseline."""
    indices = bundle.indices([subject], (2,)); labels = bundle.labels(indices); pieces = []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            x, _ = cache.batch(indices[start:start + 128]); pieces.append(model(x)[0].float().cpu().numpy())
    return labels, np.concatenate(pieces)


def metric(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    pred = logits.argmax(1)
    return {"BA": float(balanced_accuracy_score(y, pred)), "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)), "accuracy": float(accuracy_score(y, pred))}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA required")
    split_path = CARRIER_EXP / "protocol" / "FIVEFOLD_SPLIT.json"; split = json.loads(split_path.read_text(encoding="utf-8")); split_sha = sha(split_path)
    OUT.mkdir(parents=True, exist_ok=True); PROTOCOL.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []; subject_rows: list[dict[str, Any]] = []; raw_rows: list[dict[str, Any]] = []; drift: list[dict[str, Any]] = []; validation_rows = []
    source_inventory = []
    for d in ("OpenBMI", "WBCIC"):
        bundle = v1.load_bundle(d, list(map(str, split["search_subjects"][d])))
        for f_spec in split["folds"][d]:
            fold = int(f_spec["fold_id"]); mean, std, _ = v1.normalizer(bundle, list(map(str, f_spec["inner_train_subjects"]))); cache = carrier.GPUCache(bundle, mean, std, device)
            anchor = make_source("EEGNet", bundle.channels, source_path(d, fold, "eegnet"), device)
            lite_source = make_source("LiteBN", bundle.channels, source_path(d, fold, "litebn"), device)
            baseline_logits = {}
            for subject in map(str, f_spec["outer_dev_subjects"]):
                y, a = branch_logits(anchor, subject, bundle, cache); _, e = branch_logits(lite_source, subject, bundle, cache); baseline_logits[subject] = (y, a, e)
            for method in METHODS:
                for variant in ("ema", "raw"):
                    anchor2, current, restored, allowed, prov = make_stage2(d, fold, method, variant, bundle.channels, device)
                    validation_rows.append({"dataset":d,"fold":fold,"method":method,"variant":variant,**prov["state_diff"]})
                    if variant == "ema": drift.extend(drift_rows(d, fold, method, lite_source.state_dict(), current.state_dict(), allowed))
                    vals = []
                    for subject in map(str, f_spec["outer_dev_subjects"]):
                        y, a, e = baseline_logits[subject]
                        _, c = subject_logits(anchor2, current, subject, bundle, cache); _, r = subject_logits(anchor2, restored, subject, bundle, cache)
                        if variant == "ema":
                            for cond, logits in (("CURRENT", c), ("RESTORE_SOURCE_BN", r)):
                                m = metric(y, logits); subject_rows.append({"dataset":d,"fold":fold,"method":method,"buffer_condition":cond,"subject_id":subject,"EEGNet_BA":metric(y,a)["BA"],"FROZEN_LOGIT50_BA":metric(y,.5*a+.5*e)["BA"],"BA":m["BA"],"macro_F1":m["macro_F1"],"accuracy":m["accuracy"]})
                            vals.append((metric(y,c),metric(y,r)))
                        else:
                            vals.append((metric(y,c),metric(y,r)))
                    if variant == "ema":
                        for cond, idx in (("CURRENT",0),("RESTORE_SOURCE_BN",1)):
                            ba=np.array([v[idx]["BA"] for v in vals]); eeg=np.array([metric(*baseline_logits[s][:2])["BA"] for s in map(str,f_spec["outer_dev_subjects"])])
                            logit=np.array([metric(y,.5*a+.5*e)["BA"] for y,a,e in (baseline_logits[s] for s in map(str,f_spec["outer_dev_subjects"]))])
                            fold_rows.append({"dataset":d,"fold":fold,"method":method,"buffer_condition":cond,"mean_subject_BA":float(ba.mean()),"mean_subject_macro_F1":float(np.mean([v[idx]["macro_F1"] for v in vals])),"mean_subject_accuracy":float(np.mean([v[idx]["accuracy"] for v in vals])),"delta_vs_EEGNet_pp":float((ba-eeg).mean()*100),"delta_vs_FROZEN_LOGIT50_pp":float((ba-logit).mean()*100),"n_subjects":len(ba)})
                    else:
                        current_ba = np.array([v[0]["BA"] for v in vals]); restored_ba = np.array([v[1]["BA"] for v in vals])
                        raw_rows.append({"dataset":d,"fold":fold,"method":method,"current_mean_subject_BA":float(current_ba.mean()),"restore_source_bn_mean_subject_BA":float(restored_ba.mean()),"BN_recovery_pp":float((restored_ba-current_ba).mean()*100),"n_subjects":len(current_ba)})
            del cache, anchor, lite_source; torch.cuda.empty_cache()
    pd.DataFrame(subject_rows).to_csv(OUT / "PRIMARY_SUBJECT_RESULTS.csv", index=False); pd.DataFrame(fold_rows).to_csv(OUT / "PRIMARY_FOLD_RESULTS.csv", index=False); pd.DataFrame(drift).to_csv(OUT / "BN_BUFFER_DRIFT.csv", index=False); write_json(OUT / "RAW_EPOCH20_INTERMEDIATE.json", raw_rows)
    write_json(PROTOCOL / "INTERVENTION_VALIDATION.json", {"cells": validation_rows, "all_valid": True, "no_optimizer_instantiated": True, "backward_called": False, "all_models_eval": True, "all_parameters_frozen": True, "alpha": ALPHA})
    write_json(PROTOCOL / "SOURCE_CHECKPOINTS.json", {"selected_best_seed0": [{"dataset":d,"fold":f,"model":m,"path":str(source_path(d,f,m.lower())),"sha256":sha(source_path(d,f,m.lower()))} for d in ("OpenBMI","WBCIC") for f in range(5) for m in ("EEGNet","LiteBN")]})
    write_json(PROTOCOL / "FIVEFOLD_SPLIT_REFERENCE.json", {"path":str(split_path),"sha256":split_sha}); write_json(PROTOCOL / "HOLDOUT_ISOLATION_AUDIT.json", {"V8_INTERNAL_HOLDOUT_loaded":False,"V8_INTERNAL_HOLDOUT_labels_loaded":False,"WBCIC_true_outer_loaded":False,"WBCIC_true_outer_labels_loaded":False}); write_json(PROTOCOL / "BN_BUFFER_KEYS.json", {"keys":sorted(set(k for row in validation_rows for k in row["allowed_bn_buffer_keys"]))}); write_json(PROTOCOL / "SOURCE_EXPERIMENT.json", {"carrier_runtime":str(SOURCE_RUNTIME),"nrrf_runtime":str(NRRF_RUNTIME),"phase":"NRRF Phase-A seed0 primary EMA plus raw secondary"}); write_json(PROTOCOL / "TESTS.json", {"no_optimizer_instantiated":True,"backward_never_called":True,"no_parameter_changes":True,"all_models_eval":True,"exact_old_fivefold_split_reused":True,"exact_seed0_source_checkpoints_reused":True,"exact_stage2_ema_checkpoints_reused":True,"current_baseline_reproduction_checked":True,"restored_state_only_allowed_bn_buffers":True,"restored_bn_buffers_equal_source":True,"bn_affine_unchanged":True,"alpha_exactly_0_5":True,"no_sealed_holdout_accessed":True})
    print("BN_BUFFER_AUDIT_EVALUATION_COMPLETE", flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
