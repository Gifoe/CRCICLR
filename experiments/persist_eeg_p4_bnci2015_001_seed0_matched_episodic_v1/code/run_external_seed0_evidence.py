#!/usr/bin/env python3
"""BNCI2015-001 seed-0 external evidence, with frozen Full checkpoints.

This is deliberately an adapter, not a second implementation of the models,
PERSIST diagnostics, P3 variants, or the rank-matched intervention.  It calls
the authoritative sources at the paths and hashes locked below.  The only
dataset-specific work is loading the existing BNCI cache and mapping its two
sessions to the existing OpenBMI two-session semantics.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(os.environ.get("BNCI_P4_ROOT", "/root/p4_bnci2015_001_seed0_matched_episodic_v1"))
TFF = Path(os.environ.get("PERSIST_TFF_ROOT", "/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK"))
BASE_SOURCE = ROOT / "code/run_seed0_matched_episodic.py"
DIAG = ROOT / "external_diagnostics"
ARCH = ROOT / "external_architecture"
RANK = ROOT / "external_rankmatched"

ABLAT_SRC = TFF / "experiments/persist_eeg_litebn_ablation_v2_seed0/code/litebn_variants_v2.py"
ABLAT_RUN = TFF / "experiments/persist_eeg_litebn_ablation_v2_seed0/code/run_litebn_ablation_v2.py"
ABLAT_PROTOCOL = TFF / "experiments/persist_eeg_litebn_ablation_v2_seed0/protocol/PROTOCOL.json"
ABLAT_DEFS = TFF / "experiments/persist_eeg_litebn_ablation_v2_seed0/protocol/VARIANT_DEFINITIONS.json"
PEEH_SRC = TFF / "experiments/persist_eeg_litebn_ablation_v2_seed0/code/run_peeh_bridge.py"
PSWA_SRC = TFF / "experiments/persist_eeg_litebn_ablation_v2_seed0/code/run_pswa_bridge.py"
RANK_SRC = TFF / "experiments/persist_eeg_sire_rankmatched_scale_collapse_v1/code/run_rankmatched_projection_control.py"
RANK_SUMMARY_SRC = TFF / "experiments/persist_eeg_sire_rankmatched_scale_collapse_v1/code/summarize_rankmatched_projection_control.py"
RANK_PROTOCOL = TFF / "experiments/persist_eeg_sire_rankmatched_scale_collapse_v1/protocol"

EXPECTED = {
    ABLAT_SRC: "ffe8cf1db4fbb72a0478a8d9f1659150624362eb4b99437db67af4b7e271f19c",
    ABLAT_RUN: "5efe23174fd6a33735b384aea5a6a2195175d99f731f46e24c9e94e55171d95d",
    ABLAT_PROTOCOL: "bbd99e7dfcac2c1e4e3cf786f0cd9f904b278e8586c47fbf12115ed561c4e3cb",
    ABLAT_DEFS: "1dbfcd2f3ef7f0890ff07e7d9e05e72966900eeaaca1e9483923e77c22f1fa19",
    PEEH_SRC: "215cc8b393cc08f21f2536d9049527156e69230a08e5261a24abd8eb2c68dcc4",
    PSWA_SRC: "7d04ff712b1eaaa4898b8ddff43b455c8e284dc706c6bd3af7a5a3a8e42c5251",
    RANK_SRC: "4a603ce9b892994c5e40ce7644e1060c3009f3e980be708600cfaedb42c2cf4e",
    RANK_SUMMARY_SRC: "e8191d229e88347893f9cdbb2254d621f2110cdd5a063fb099e06a62e31cb93a",
}
VARIANTS = ("B1_SAME_SCALE_63", "B2_SCALE_COLLAPSE", "B3_SINGLE_SPATIAL_BASIS", "B4_ONE_STAGE_BACKEND")
VARIANT_COUNTS = {"B1_SAME_SCALE_63": 45498, "B2_SCALE_COLLAPSE": 45626,
                  "B3_SINGLE_SPATIAL_BASIS": 45698, "B4_ONE_STAGE_BACKEND": 39418}
SEED = 0


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def text_write(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def csv_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([clean(row) for row in rows])
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def direct_import(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_audit() -> dict[str, Any]:
    observed = {str(path): sha(path) if path.is_file() else None for path in EXPECTED}
    bad = {path: (EXPECTED[Path(path)], value) for path, value in observed.items() if value != EXPECTED[Path(path)]}
    if bad:
        raise RuntimeError(f"authoritative source hash mismatch: {bad}")
    freeze_path = RANK_PROTOCOL / "PROJECTION_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    arrays, manifest = RANK_PROTOCOL / "random_projectors.npz", RANK_PROTOCOL / "RANDOM_PROJECTOR_MANIFEST.csv"
    if sha(arrays) != freeze["arrays_sha256"] or sha(manifest) != freeze["manifest_sha256"]:
        raise RuntimeError("frozen rank-matched projector bank drift")
    return {"tff_worktree": str(TFF), "tff_commit": "60d053f0cf1c6d8f77a0fd337f0ca8ad03224843",
            "source_hashes": observed, "rank_projector_freeze": {"path": str(freeze_path), **freeze},
            "rank_projector_arrays_path": str(arrays), "rank_projector_manifest_path": str(manifest)}


def common():
    audit = source_audit()
    base = direct_import("bnci_frozen_base", BASE_SOURCE)
    stage1, eegnet, sire, recipe = base.load_authoritative()
    full_audit = base.audit_models(eegnet, sire)
    bundle, raw = base.load_bundle(stage1)
    folds, _ = base.make_splits()
    frozen = json.loads((ROOT / "seed0_training_records.json").read_text(encoding="utf-8"))
    records = {(row["model"], int(row["fold"])): row for row in frozen["records"]}
    if len(records) != 10 or any((name, fold["fold_id"]) not in records for name in ("EEGNet", "SIRE-EEG") for fold in folds):
        raise RuntimeError("frozen training record coverage failure")
    normalizers, manifests, paths, manifest_audits = {}, {}, {}, {}
    for fold in folds:
        fid = int(fold["fold_id"])
        mean, std, norm = base.normalize(raw, bundle, fold["inner_train_subjects"])
        full_record = records[("SIRE-EEG", fid)]
        if norm["mean_std_sha256"] != full_record["normalizer_sha256"]:
            raise RuntimeError(f"frozen S1 normalizer mismatch f{fid}")
        manifest, maudit, mpath = base.build_manifest(stage1, bundle, fold)
        if sha(mpath) != full_record["manifest_sha256"]:
            raise RuntimeError(f"frozen authoritative episode manifest mismatch f{fid}")
        normalizers[fid] = (mean, std, norm)
        manifests[fid], paths[fid], manifest_audits[fid] = manifest, mpath, maudit
    return base, eegnet, sire, bundle, raw, folds, records, normalizers, manifests, paths, manifest_audits, recipe, full_audit, audit


def metadata(bundle, indices: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"subject_id": [str(bundle.search_rows[int(i)].subject) for i in indices],
                         "session_id": [int(bundle.search_rows[int(i)].session) for i in indices],
                         "label": bundle.labels(indices).astype(np.int64)})


def extract(model, cache, indices: np.ndarray) -> np.ndarray:
    values = []
    model.eval()
    with torch.inference_mode():
        for lo in range(0, len(indices), 128):
            x, _ = cache.batch(indices[lo:lo + 128])
            output = model(x)
            if not isinstance(output, tuple) or len(output) != 2 or tuple(output[1].shape[1:]) != (64,):
                raise RuntimeError("frozen model does not expose an exact 64-D pre-classifier embedding")
            values.append(output[1].float().cpu().numpy())
    h = np.concatenate(values).astype(np.float32)
    if not np.isfinite(h).all():
        raise RuntimeError("non-finite frozen embedding")
    return h


def frozen_model(base, eegnet, sire, name: str, record: Mapping[str, Any], device: torch.device):
    model = base.constructor(name, eegnet, sire).to(device)
    ckpt = Path(record["checkpoint_path"])
    if not ckpt.is_file() or sha(ckpt) != record["checkpoint_sha256"]:
        raise RuntimeError(f"frozen checkpoint provenance mismatch: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False), strict=True)
    model.eval()
    if any(item.training for item in model.modules()):
        raise RuntimeError("frozen model eval lock failed")
    return model


def empty_peeh(result: dict[str, Any]) -> dict[str, Any]:
    """Paper convention: an empty Protected assignment is exactly zero effect."""
    if int(result["protected_rank"]) != 0:
        return result
    for row in result["subject_rows"]:
        intact = float(row["intact_probe_BA"])
        row.update({"protected_erased_BA": intact, "random_erased_BA": intact,
                    "protected_harm_pp": 0.0, "random_harm_pp": 0.0, "PEEH_pp": 0.0})
    result.update({"protected_erased_BA": float(result["intact_probe_BA"]),
                   "random_erased_BA": float(result["intact_probe_BA"]),
                   "protected_harm_pp": 0.0, "random_harm_pp": 0.0, "PEEH_pp": 0.0,
                   "empty_protected_effect_fixed_to_zero": True})
    return result


def pswa_for_fold(pswa, peeh_result: Mapping[str, Any], train_h: np.ndarray, train_meta: pd.DataFrame,
                  eval_h: np.ndarray, eval_meta: pd.DataFrame, outer: Sequence[str], model: str, fold: int) -> dict[str, Any]:
    protected = list(map(int, peeh_result["protected_coordinates"]))
    rank, active = len(protected), int(peeh_result["active_rank"])
    base_row = {"model": model, "fold": fold, "seed": 0, "protected_rank": rank, "active_rank": active,
                "protected_coordinates": protected}
    if rank == 0:
        return {**base_row, "status": "EMPTY_PROTECTED_UNDEFINED_OMITTED", "subject_rows": [], "random_sets": []}
    basis = pswa.canonical_basis(train_h, train_meta.subject_id.to_numpy(), train_meta.session_id.to_numpy(),
                                 train_meta.label.to_numpy(), peeh_result)
    ztrain, zeval = pswa.z_coordinates(train_h, basis), pswa.z_coordinates(eval_h, basis)
    ytrain, yeval = train_meta.label.to_numpy(int), eval_meta.label.to_numpy(int)
    ppack = pswa.ridge_fit(ztrain[:, protected], ytrain, 2)
    ppred = pswa.ridge_predict(zeval[:, protected], ppack)
    random_sets = pswa.random_sets("BNCI2015_001", model, fold, 0, rank, active)
    random_preds = [pswa.ridge_predict(zeval[:, coords], pswa.ridge_fit(ztrain[:, coords], ytrain, 2)) for coords in random_sets]
    rows = []
    for subject in sorted(map(str, outer)):
        pba, rba = {}, {"S1": [], "S2": []}
        for session, label in ((1, "S1"), (2, "S2")):
            mask = (eval_meta.subject_id.astype(str).to_numpy() == subject) & (eval_meta.session_id.to_numpy(int) == session)
            if int(mask.sum()) != 200:
                raise RuntimeError(f"PSWA outer cell mismatch f{fold}/{subject}/{label}")
            pba[label] = pswa.ba(yeval[mask], ppred[mask], 2)
            rba[label] = [pswa.ba(yeval[mask], pred[mask], 2) for pred in random_preds]
        pws = min(pba.values())
        rws = np.minimum(np.asarray(rba["S1"]), np.asarray(rba["S2"]))
        rows.append({**base_row, "subject": subject, "protected_BA_S1": pba["S1"], "protected_BA_S2": pba["S2"],
                     "random_BA_S1": float(np.mean(rba["S1"])), "random_BA_S2": float(np.mean(rba["S2"])),
                     "protected_only_WSBA": pws, "random_only_WSBA": float(rws.mean()),
                     "PSWA_pp": float(100 * (pws - rws.mean())), "random_draws": len(random_sets)})
    return {**base_row, "status": "VALID", "subject_rows": rows, "random_sets": random_sets,
            "basis_dimensions": int(len(basis["rho"]))}


def run_diagnostics() -> None:
    base, eegnet, sire, bundle, raw, folds, records, normals, *_rest = common()
    audit = _rest[-1]
    peeh = direct_import("bnci_external_peeh", PEEH_SRC)
    pswa = direct_import("bnci_external_pswa", PSWA_SRC)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_runs, peeh_rows, pswa_runs, pswa_rows = [], [], [], []
    for fold_info in folds:
        fold = int(fold_info["fold_id"]); mean, std, norm = normals[fold]
        cache = base.GPUCache(raw, bundle, mean, std, device)
        train_idx = bundle.indices(fold_info["inner_train_subjects"], (1, 2))
        eval_s2_idx = bundle.indices(fold_info["outer_test_subjects"], (2,))
        eval_all_idx = bundle.indices(fold_info["outer_test_subjects"], (1, 2))
        train_meta, eval_s2_meta, eval_all_meta = metadata(bundle, train_idx), metadata(bundle, eval_s2_idx), metadata(bundle, eval_all_idx)
        for name in ("EEGNet", "SIRE-EEG"):
            record = records[(name, fold)]
            model = frozen_model(base, eegnet, sire, name, record, device)
            train_h, eval_s2_h, eval_all_h = extract(model, cache, train_idx), extract(model, cache, eval_s2_idx), extract(model, cache, eval_all_idx)
            emb = {"train_h": train_h, "train_y": train_meta.label.to_numpy(int), "train_subject": train_meta.subject_id.astype(str).to_numpy(),
                   "train_session": train_meta.session_id.to_numpy(int), "eval_h": eval_s2_h, "eval_y": eval_s2_meta.label.to_numpy(int),
                   "eval_subject": eval_s2_meta.subject_id.astype(str).to_numpy(), "eval_session": eval_s2_meta.session_id.to_numpy(int),
                   "metadata": {"checkpoint": record["checkpoint_path"], "checkpoint_sha256": record["checkpoint_sha256"],
                                "normalizer_sha256": norm["mean_std_sha256"], "representation": "64-D classifier input; eval mode"}}
            run = empty_peeh(peeh.run_one(emb, "BNCI2015_001", name, fold, 0, 2))
            run["outer_sessions_for_peeh"] = ["S2=1B"]
            json_write(DIAG / "runtime/peeh_runs" / name / f"fold{fold}_seed0.json", run)
            result_runs.append(run)
            for row in run["subject_rows"]:
                peeh_rows.append({**row, "checkpoint_sha256": record["checkpoint_sha256"], "normalizer_sha256": norm["mean_std_sha256"],
                                  "empty_protected_effect_fixed_to_zero": int(run["protected_rank"]) == 0})
            pswa_run = pswa_for_fold(pswa, run, train_h, train_meta, eval_all_h, eval_all_meta, fold_info["outer_test_subjects"], name, fold)
            json_write(DIAG / "runtime/pswa_runs" / name / f"fold{fold}_seed0.json", pswa_run)
            pswa_runs.append(pswa_run)
            for row in pswa_run["subject_rows"]:
                pswa_rows.append({**row, "checkpoint_sha256": record["checkpoint_sha256"], "normalizer_sha256": norm["mean_std_sha256"]})
            del model
        del cache
        if device.type == "cuda": torch.cuda.empty_cache()
    csv_write(DIAG / "seed0_peeh_subject_results.csv", peeh_rows)
    csv_write(DIAG / "seed0_pswa_subject_results.csv", pswa_rows)
    assignments = {f"{run['model']}/fold{run['fold']}": {"protected_coordinates": run["protected_coordinates"],
                    "protected_rank": run["protected_rank"], "active_rank": run["active_rank"],
                    "persistence_audit": run["spectrum_audit"], "selection_blocks": run["block_selection"],
                    "pswa_status": next(item["status"] for item in pswa_runs if item["model"] == run["model"] and item["fold"] == run["fold"])} for run in result_runs}
    json_write(DIAG / "diagnostic_assignments.json", assignments)
    summary = []
    for name in ("EEGNet", "SIRE-EEG"):
        pruns = [item for item in result_runs if item["model"] == name]
        values = np.asarray([row["PEEH_pp"] for row in peeh_rows if row["model"] == name], float)
        boot = peeh.bootstrap(values, peeh.stable_seed("BNCI2015-001", "PEEH", name), peeh.FINAL_BOOTSTRAP_DRAWS)
        ranks = [int(item["protected_rank"]) for item in pruns]
        group = pd.DataFrame([row for row in peeh_rows if row["model"] == name])
        summary.append({"diagnostic": "PEEH", "model": name, "biological_subjects": len(group), "mean_pp": boot["mean"],
                        "CI95_low_pp": boot["ci95"][0], "CI95_high_pp": boot["ci95"][1], "protected_ranks_by_fold": ranks,
                        "nonempty_coverage": f"{sum(rank > 0 for rank in ranks)}/5", "intact_ridge_BA": float(group.intact_probe_BA.mean()),
                        "matched_rank_random_control_BA": float(group.random_erased_BA.mean()), "protected_ridge_BA": float(group.protected_erased_BA.mean()),
                        "bootstrap_draws": peeh.FINAL_BOOTSTRAP_DRAWS, "empty_protected_convention": "exactly 0 PEEH effect"})
        sruns = [item for item in pswa_runs if item["model"] == name]
        srows = [row for row in pswa_rows if row["model"] == name]
        sranks = [int(item["protected_rank"]) for item in sruns]
        if srows:
            svals = np.asarray([row["PSWA_pp"] for row in srows], float)
            low, high = pswa.bootstrap(svals, pswa.stable_seed("BNCI2015-001", "PSWA", name))
            mean_pp, pba, rba = float(svals.mean()), float(np.mean([row["protected_only_WSBA"] for row in srows])), float(np.mean([row["random_only_WSBA"] for row in srows]))
        else:
            mean_pp = low = high = pba = rba = None
        summary.append({"diagnostic": "PSWA", "model": name, "biological_subjects": len(srows), "mean_pp": mean_pp,
                        "CI95_low_pp": low, "CI95_high_pp": high, "protected_ranks_by_fold": sranks,
                        "nonempty_coverage": f"{sum(rank > 0 for rank in sranks)}/5", "intact_ridge_BA": None,
                        "matched_rank_random_control_BA": rba, "protected_ridge_BA": pba, "bootstrap_draws": pswa.BOOTSTRAP_DRAWS,
                        "empty_protected_convention": "undefined and omitted"})
    csv_write(DIAG / "seed0_diagnostic_summary.csv", summary)
    text_write(DIAG / "PERSIST_DIAGNOSTIC_PROTOCOL.md", ["# BNCI2015-001 frozen PERSIST diagnostics", "",
        "No neural model was trained, adapted, or selected in this phase. S1=0A and S2=1B form the exact two-session construction.",
        "Protected selection, whitening and persistence use only each fold's inner-train subjects from both sessions. PEEH is evaluated on the disjoint outer S2 subjects. PSWA uses outer S1/S2 and is omitted for empty Protected assignments.",
        f"Constants imported without alteration: 200 persistence permutations, 100 equal-rank controls, ridge alpha {peeh.RIDGE_ALPHA}, {peeh.FINAL_BOOTSTRAP_DRAWS} biological-subject bootstrap draws."])
    text_write(DIAG / "DIAGNOSTIC_IMPLEMENTATION_AUDIT.md", ["# Diagnostic implementation audit", "",
        f"- PEEH source: `{PEEH_SRC}` SHA-256 `{EXPECTED[PEEH_SRC]}`.", f"- PSWA source: `{PSWA_SRC}` SHA-256 `{EXPECTED[PSWA_SRC]}`.",
        "- The adapter invokes `run_one`, `build_spectrum`, `select_protected`, `evaluate_erasure`, `canonical_basis`, `z_coordinates`, `random_sets`, and ridge probes from those exact sources; it does not recode their rules.",
        "- Both models expose the eval-mode 64-D tensor immediately before the classifier. Frozen checkpoint and S1-normalizer hashes are recorded per output row.",
        "- Empty Protected gives exactly zero PEEH; PSWA does not manufacture a zero or a random control for an empty assignment. Coverage is nonempty fold/checkpoint coverage out of 5."])
    lines = ["# BNCI2015-001 seed-0 PERSIST diagnostic summary", "", "PEEH/PSWA are diagnostic quantities, not classifier-quality scores.", "",
             "| Diagnostic | Model | Mean pp [95% CI] | Protected ranks by fold | Coverage | Intact/protected or random probe BA |", "|---|---|---:|---|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['diagnostic']} | {row['model']} | {row['mean_pp'] if row['mean_pp'] is not None else 'undefined'} {('[' + str(row['CI95_low_pp']) + ', ' + str(row['CI95_high_pp']) + ']') if row['mean_pp'] is not None else ''} | {row['protected_ranks_by_fold']} | {row['nonempty_coverage']} | {row['intact_ridge_BA'] if row['diagnostic']=='PEEH' else row['protected_ridge_BA']} / {row['matched_rank_random_control_BA']} |")
    text_write(DIAG / "SEED0_DIAGNOSTIC_SUMMARY.md", lines)
    json_write(DIAG / "diagnostic_execution_audit.json", {"source_audit": audit, "device": str(device), "runs": len(result_runs), "no_training": True})
    print("BNCI_EXTERNAL_DIAGNOSTICS_COMPLETE", flush=True)


def state_shape_audit(model) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in model.state_dict().items() if not key.endswith("num_batches_tracked")}


def train_variant(base, variants, variant: str, fold_info: Mapping[str, Any], manifest, manifest_path: Path, bundle, cache, norm: Mapping[str, Any]) -> dict[str, Any]:
    fold = int(fold_info["fold_id"])
    base.set_seed(base.SEED)
    model = variants.LiteBNAblationV2(13, 2, variant).to(cache.device)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != VARIANT_COUNTS[variant]:
        raise RuntimeError(f"P3 parameter audit failed {variant}: {count}")
    x = torch.zeros((1, 13, 1000), device=cache.device)
    with torch.inference_mode():
        logits, h = model(x)
    if tuple(logits.shape) != (1, 2) or tuple(h.shape) != (1, 64):
        raise RuntimeError(f"P3 forward audit failed {variant}")
    base.set_seed(base.TRAINING_SEED)
    dest = ARCH / "runtime/checkpoints" / variant / f"fold{fold}_seed0"; dest.mkdir(parents=True, exist_ok=True)
    latest, selected = dest / "checkpoint_latest.pt", dest / "selected_best.pt"
    init_sha, manifest_sha = base.state_hash(copy.deepcopy(model.state_dict())), sha(manifest_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    amp = cache.device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, history, best, best_epoch, best_state = 1, [], -float("inf"), None, None
    if latest.exists():
        saved = torch.load(latest, map_location=cache.device, weights_only=False)
        checks = {"variant": variant, "init_sha256": init_sha, "manifest_sha256": manifest_sha, "normalizer_sha256": norm["mean_std_sha256"]}
        if any(saved.get(k) != v for k, v in checks.items()):
            raise RuntimeError(f"unsafe ablation resume {variant}/f{fold}")
        model.load_state_dict(saved["current_state"]); optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"]); base.restore_rng(saved["rng"])
        start, history, best, best_epoch, best_state = int(saved["epoch"]) + 1, saved["history"], saved["best_val_BA"], saved["best_epoch"], saved["best_state"]
    started = time.perf_counter()
    for epoch in range(start, base.EPOCHS + 1):
        model.train(); losses = []
        for episode in manifest[epoch - 1]:
            x, y = cache.batch(episode["support_indices"] + episode["query_indices"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=cache.device.type, dtype=torch.float16, enabled=amp):
                out, _ = model(x); loss = F.cross_entropy(out, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite CE {variant}/f{fold}")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), base.GRADIENT_CLIP); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        validation = base.evaluate_session(model, bundle, cache, fold_info["inner_val_subjects"], 2)
        val_ba = float(np.mean([one["BA"] for one in validation.values()])); chosen = epoch >= base.MIN_EPOCH and val_ba > best + 1e-12
        if chosen:
            best, best_epoch, best_state = val_ba, epoch, copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "CE": float(np.mean(losses)), "inner_val_S2_subject_BA": val_ba, "selected": bool(chosen)})
        torch.save({"variant": variant, "epoch": epoch, "history": history, "best_val_BA": best, "best_epoch": best_epoch, "best_state": best_state,
                    "current_state": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": base.rng_state(),
                    "manifest_sha256": manifest_sha, "init_sha256": init_sha, "normalizer_sha256": norm["mean_std_sha256"]}, latest)
        if epoch == 1 or epoch % 5 == 0 or chosen:
            print(f"[{variant} f{fold}] epoch={epoch:02d} CE={history[-1]['CE']:.4f} valS2BA={val_ba:.4f}", flush=True)
    if best_state is None or best_epoch is None:
        raise RuntimeError(f"no eligible P3 checkpoint {variant}/f{fold}")
    model.load_state_dict(best_state); torch.save(model.state_dict(), selected)
    record = {"model": variant, "fold": fold, "seed": 0, "selected_epoch": int(best_epoch), "best_inner_val_S2_subject_BA": float(best),
              "checkpoint_path": str(selected), "checkpoint_sha256": sha(selected), "latest_checkpoint_path": str(latest), "init_sha256": init_sha,
              "manifest_sha256": manifest_sha, "normalizer_sha256": norm["mean_std_sha256"], "parameter_count": count, "amp": amp,
              "elapsed_seconds": time.perf_counter() - started, "history": history, "inner_train_subjects": fold_info["inner_train_subjects"],
              "inner_val_subjects": fold_info["inner_val_subjects"], "outer_test_subjects": fold_info["outer_test_subjects"],
              "variant_source": str(ABLAT_SRC), "variant_source_sha256": EXPECTED[ABLAT_SRC]}
    del model
    if cache.device.type == "cuda": torch.cuda.empty_cache()
    return record


def subject_summary(rows: Sequence[Mapping[str, Any]], model: str, subjects: Sequence[str]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    result = {}
    for subject in subjects:
        part = [row for row in rows if row["model"] == model and row["subject"] == subject]
        sess = {row["session"]: row for row in part}
        if set(sess) != {"S1", "S2"}:
            raise RuntimeError(f"ablation subject/session coverage {model}/{subject}")
        result[subject] = {"future_S2_BA": float(sess["S2"]["BA"]), "future_S2_Macro_F1": float(sess["S2"]["Macro_F1"]),
                           "WS_BA": min(float(sess["S1"]["BA"]), float(sess["S2"]["BA"]))}
    return {key: float(np.mean([result[s][key] for s in subjects])) for key in ("future_S2_BA", "future_S2_Macro_F1", "WS_BA")}, result


def run_architecture() -> None:
    base, eegnet, sire, bundle, raw, folds, records, normals, manifests, paths, audits, recipe, full_audit, source = common()
    variants = direct_import("bnci_p3_variants", ABLAT_SRC)
    if tuple(variants.LiteBNAblationV2.VARIANTS) != VARIANTS:
        raise RuntimeError("P3 variant registry mismatch")
    b2 = variants.LiteBNAblationV2(13, 2, "B2_SCALE_COLLAPSE")
    full = sire.CompactLite(13, "bn")
    if state_shape_audit(b2) != state_shape_audit(full):
        raise RuntimeError("P3 B2 does not have the traced Full tensor structure")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained = []
    for fold_info in folds:
        fold = int(fold_info["fold_id"]); mean, std, norm = normals[fold]; cache = base.GPUCache(raw, bundle, mean, std, device)
        for variant in VARIANTS:
            trained.append(train_variant(base, variants, variant, fold_info, manifests[fold], paths[fold], bundle, cache, norm))
        del cache
        if device.type == "cuda": torch.cuda.empty_cache()
    frozen_subject = pd.read_csv(ROOT / "seed0_subject_metrics.csv")
    rows = []
    for row in frozen_subject[frozen_subject.model == "SIRE-EEG"].to_dict("records"):
        rows.append({"model": "B0_FULL", "fold": int(row["fold"]), "seed": 0, "subject": row["subject"], "session": row["session"],
                     "BA": float(row["BA"]), "Macro_F1": float(row["Macro_F1"]), "trials": int(row["trials"]), "selected_epoch": int(row["selected_epoch"]),
                     "checkpoint_sha256": row["checkpoint_sha256"], "parameter_count": 45626, "frozen_reused": True})
    index = {(row["model"], int(row["fold"])): row for row in trained}
    outer_fold = {}
    for fold_info in folds:
        fold = int(fold_info["fold_id"]); mean, std, _ = normals[fold]; cache = base.GPUCache(raw, bundle, mean, std, device)
        for variant in VARIANTS:
            record = index[(variant, fold)]; model = variants.LiteBNAblationV2(13, 2, variant).to(device)
            model.load_state_dict(torch.load(record["checkpoint_path"], map_location=device, weights_only=False), strict=True); model.eval()
            for session, sid in (("S1", 1), ("S2", 2)):
                values = base.evaluate_session(model, bundle, cache, fold_info["outer_test_subjects"], sid)
                if session == "S2": outer_fold[(variant, fold)] = float(np.mean([item["BA"] for item in values.values()]))
                for subject_id, metric in values.items():
                    rows.append({"model": variant, "fold": fold, "seed": 0, "subject": subject_id, "session": session, **metric,
                                 "trials": 200, "selected_epoch": record["selected_epoch"], "checkpoint_sha256": record["checkpoint_sha256"],
                                 "parameter_count": record["parameter_count"], "frozen_reused": False})
            del model
        del cache
        if device.type == "cuda": torch.cuda.empty_cache()
    csv_write(ARCH / "seed0_ablation_subject_metrics.csv", rows)
    subjects = tuple(base.SUBJECTS); stats, cells = {}, {}
    for model in ("B0_FULL",) + VARIANTS:
        stats[model], cells[model] = subject_summary(rows, model, subjects)
    summary = []
    full_records = [records[("SIRE-EEG", int(f["fold_id"]))] for f in folds]
    for model in ("B0_FULL",) + VARIANTS:
        model_records = full_records if model == "B0_FULL" else [index[(model, int(f["fold_id"]))] for f in folds]
        summary.append({"row_type": "model", "model": model, **stats[model], "parameter_count": 45626 if model == "B0_FULL" else VARIANT_COUNTS[model],
                        "frozen_reused": model == "B0_FULL", "paired_metric": None, "paired_delta_mean": None, "CI95_low": None, "CI95_high": None,
                        "improved_subjects": None, "tied_subjects": None, "harmed_subjects": None})
        for record in model_records:
            f = int(record["fold"])
            future = float(np.mean([cells[model][s]["future_S2_BA"] for s in record["outer_test_subjects"]]))
            summary.append({"row_type": "fold", "model": model, "fold": f, "future_S2_BA": future, "future_S2_Macro_F1": None, "WS_BA": None,
                            "parameter_count": record["parameter_count"], "frozen_reused": model == "B0_FULL", "selected_epoch": record["selected_epoch"],
                            "inner_val_S2_BA": record["best_inner_val_S2_subject_BA"], "outer_S2_BA": future})
    for model in VARIANTS:
        for metric in ("future_S2_BA", "WS_BA"):
            delta = np.asarray([cells[model][s][metric] - cells["B0_FULL"][s][metric] for s in subjects], float)
            mean, lo, hi = base.bootstrap(delta, f"BNCI2015-001-{model}-{metric}")
            summary.append({"row_type": "paired_contrast", "model": model, "paired_metric": metric, "paired_delta_mean": mean, "CI95_low": lo, "CI95_high": hi,
                            "improved_subjects": int(np.sum(delta > 1e-12)), "tied_subjects": int(np.sum(np.abs(delta) <= 1e-12)), "harmed_subjects": int(np.sum(delta < -1e-12)),
                            "bootstrap_draws": 20000})
    csv_write(ARCH / "seed0_ablation_summary.csv", summary)
    json_write(ARCH / "seed0_ablation_training_records.json", {"seed": 0, "B0_FULL": {"frozen_reused": True, "records": full_records},
               "B1_to_B4": trained, "recipe": recipe, "variant_counts": VARIANT_COUNTS, "source_audit": source})
    text_write(ARCH / "ARCHITECTURE_PROVENANCE_AUDIT.md", ["# External architecture provenance audit", "",
        f"- Frozen B0 Full: final `CompactLite(13, 'bn')`, source `{base.SIRE_SOURCE}` SHA-256 `{base.EXPECTED_SHA256[base.SIRE_SOURCE]}`, 45,626 parameters. Its existing BNCI selected checkpoints were reused without loading an optimizer or training step.",
        f"- B1--B4 implementation: `{ABLAT_SRC}` SHA-256 `{EXPECTED[ABLAT_SRC]}`, in P3 worktree commit `60d053f0cf1c6d8f77a0fd337f0ca8ad03224843`.",
        f"- P3 training wrapper/config traced at `{ABLAT_RUN}` SHA-256 `{EXPECTED[ABLAT_RUN]}` and `{ABLAT_PROTOCOL}` SHA-256 `{EXPECTED[ABLAT_PROTOCOL]}`.",
        "- B2 has exactly the same C=13 state-tensor key/shape map as Full. B1/B3/B4 were instantiated directly from the P3 class and audited for count and [1,2]/[1,64] forward outputs before training.",
        "- Counts: B1=45,498; B2=45,626; B3=45,698; B4=39,418. No architecture was changed after BNCI outcomes."])
    text_write(ARCH / "ABLATION_PROTOCOL.md", ["# BNCI2015-001 P3 architecture replication", "",
        "B0 Full is a frozen reuse. Only B1 SameScale-63, B2 ScaleCollapse, B3 SingleSpatialBasis, and B4 OneStageBackend are trained.",
        "Every B1--B4 fold reuses the frozen full fold's cached [13,1000] tensors, S1-only inner-train normalizer, exact 60x20 authoritative episode manifest, split, AdamW lr=3e-4, weight decay=5e-4, ordinary CE, clip=5, AMP convention, 60 epochs and S2 inner-validation earliest-tie checkpoint rule.",
        "Support is 64 S1 trials from four inner-train subjects; query is 64 S2 trials from four different inner-train subjects. Validation and outer subjects never enter an episode."])
    lines = ["# BNCI2015-001 seed-0 architecture ablation summary", "", "This is descriptive external replication; it does not select an architecture.", "",
             "| Model | Future S2 BA | Future S2 Macro-F1 | WS-BA | Parameters |", "|---|---:|---:|---:|---:|"]
    for model in ("B0_FULL",) + VARIANTS:
        item = stats[model]; lines.append(f"| {model} | {item['future_S2_BA']:.4f} | {item['future_S2_Macro_F1']:.4f} | {item['WS_BA']:.4f} | {45626 if model=='B0_FULL' else VARIANT_COUNTS[model]:,} |")
    lines += ["", "| Variant - Full | Metric | Mean delta | 95% CI | Improved / tied / harmed subjects |", "|---|---|---:|---:|---:|"]
    for row in summary:
        if row["row_type"] == "paired_contrast": lines.append(f"| {row['model']} | {row['paired_metric']} | {row['paired_delta_mean']:+.4f} | [{row['CI95_low']:+.4f}, {row['CI95_high']:+.4f}] | {row['improved_subjects']} / {row['tied_subjects']} / {row['harmed_subjects']} |")
    text_write(ARCH / "SEED0_ABLATION_SUMMARY.md", lines)
    print("BNCI_EXTERNAL_ARCHITECTURE_COMPLETE", flush=True)


def metrics(base, labels: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    return float(base.balanced_accuracy_score(labels, prediction)), float(base.f1_score(labels, prediction, average="macro"))


def run_rankmatched() -> None:
    base, eegnet, sire, bundle, raw, folds, records, normals, *_rest = common()
    source = _rest[-1]
    rank = direct_import("bnci_rankmatched", RANK_SRC)
    rank_summary = direct_import("bnci_rankmatched_summary", RANK_SUMMARY_SRC)
    frozen = pd.read_csv(ROOT / "seed0_subject_metrics.csv")
    expected = {(int(row.fold), str(row.subject), str(row.session)): (float(row.BA), float(row.Macro_F1)) for row in frozen[frozen.model == "SIRE-EEG"].itertuples(index=False)}
    if len(expected) != 24:
        raise RuntimeError("frozen Full outer-cell coverage mismatch")
    with np.load(RANK_PROTOCOL / "random_projectors.npz", allow_pickle=False) as bank:
        matrices_np = np.concatenate((bank["identity"][None], bank["scale"][None], bank["projectors"])).astype(np.float32)
    if matrices_np.shape != (102, 48, 48):
        raise RuntimeError("authoritative projector bank shape mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay, projected_cells = [], []
    # Replay is intentionally completed and checked before a single projector outcome is evaluated.
    for fold_info in folds:
        fold = int(fold_info["fold_id"]); mean, std, norm = normals[fold]; cache = base.GPUCache(raw, bundle, mean, std, device)
        record = records[("SIRE-EEG", fold)]; model = frozen_model(base, eegnet, sire, "SIRE-EEG", record, device)
        for session, sid in (("S1", 1), ("S2", 2)):
            got = base.evaluate_session(model, bundle, cache, fold_info["outer_test_subjects"], sid)
            for subject, item in got.items():
                want_ba, want_f1 = expected[(fold, subject, session)]
                diff_ba, diff_f1 = abs(item["BA"] - want_ba), abs(item["Macro_F1"] - want_f1)
                replay.append({"fold": fold, "subject": subject, "session": session, "replayed_BA": item["BA"], "frozen_BA": want_ba,
                               "replayed_Macro_F1": item["Macro_F1"], "frozen_Macro_F1": want_f1, "abs_BA_difference": diff_ba,
                               "abs_Macro_F1_difference": diff_f1, "pass": diff_ba <= 1e-7 and diff_f1 <= 1e-7,
                               "checkpoint_sha256": record["checkpoint_sha256"], "normalizer_sha256": norm["mean_std_sha256"]})
        del model, cache
        if device.type == "cuda": torch.cuda.empty_cache()
    csv_write(RANK / "runtime/replay_cells.csv", replay)
    if not all(row["pass"] for row in replay):
        json_write(RANK / "runtime/replay_failure.json", {"rows": replay, "tolerance": 1e-7, "intervention_started": False})
        raise RuntimeError("Full BNCI replay mismatch; rank-matched intervention prohibited")
    matrices = torch.as_tensor(matrices_np, device=device)
    identity_max = 0.0
    for fold_info in folds:
        fold = int(fold_info["fold_id"]); mean, std, norm = normals[fold]; cache = base.GPUCache(raw, bundle, mean, std, device)
        record = records[("SIRE-EEG", fold)]; model = frozen_model(base, eegnet, sire, "SIRE-EEG", record, device)
        with torch.inference_mode():
            for subject in fold_info["outer_test_subjects"]:
                for session, sid in (("S1", 1), ("S2", 2)):
                    ids = bundle.indices([subject], (sid,)); y = bundle.labels(ids); predictions = [[] for _ in range(102)]
                    for lo in range(0, len(ids), 32):
                        x, _ = cache.batch(ids[lo:lo + 32]); z = rank.branch_output(model, x)
                        first = rank.projected_logits(model, z, matrices[:2]); original = model(x)[0]
                        err = float((first[0] - original).abs().max()); identity_max = max(identity_max, err)
                        if err >= 1e-5: raise RuntimeError(f"identity wrapper mismatch f{fold}/{subject}/{session}: {err}")
                        predictions[0].append(first[0].argmax(-1).cpu().numpy()); predictions[1].append(first[1].argmax(-1).cpu().numpy())
                        for start in range(2, 102, 20):
                            logits = rank.projected_logits(model, z, matrices[start:start + 20])
                            for off, pred in enumerate(logits.argmax(-1).cpu().numpy()): predictions[start + off].append(pred)
                    for j, chunks in enumerate(predictions):
                        ba, f1 = metrics(base, y, np.concatenate(chunks))
                        projected_cells.append({"fold": fold, "seed": 0, "subject": subject, "session": session,
                                                "condition": "Intact" if j == 0 else ("ScaleCollapse" if j == 1 else "RandomRank16"),
                                                "projector_id": -1 if j < 2 else j - 2, "BA": ba, "Macro_F1": f1, "trials": len(y),
                                                "checkpoint_sha256": record["checkpoint_sha256"], "normalizer_sha256": norm["mean_std_sha256"]})
        del model, cache
        if device.type == "cuda": torch.cuda.empty_cache()
    if identity_max >= 1e-5:
        raise RuntimeError("identity intervention wrapper audit failed")
    csv_write(RANK / "runtime/projected_cells.csv", projected_cells)
    rows, projector_effects = [], []
    frame = pd.DataFrame(projected_cells)
    for subject in base.SUBJECTS:
        part = frame[frame.subject == subject]
        one = {}
        for condition in ("Intact", "ScaleCollapse"):
            sessions = part[part.condition == condition].set_index("session")
            one[condition] = {"BA": float(sessions.loc["S2", "BA"]), "WS_BA": float(sessions.BA.min())}
        rnd = part[part.condition == "RandomRank16"]
        for metric in ("BA", "WS_BA"):
            if metric == "BA": random_values = rnd[rnd.session == "S2"].set_index("projector_id").BA.sort_index().to_numpy(float)
            else:
                random_values = rnd.pivot(index="projector_id", columns="session", values="BA")[["S1", "S2"]].min(axis=1).sort_index().to_numpy(float)
            if len(random_values) != 100: raise RuntimeError(f"random projector count mismatch for {subject}/{metric}")
            for pid, value in enumerate(random_values): projector_effects.append({"subject": subject, "metric": "future_S2_BA" if metric == "BA" else metric, "projector_id": pid, "random_metric": float(value), "random_harm_pp": 100 * (one["Intact"][metric] - value)})
            one[f"random_{metric}"] = float(random_values.mean())
        rows.append({"subject": subject, "fold": int(part.fold.iloc[0]), "intact_BA": one["Intact"]["BA"], "ScaleCollapse_BA": one["ScaleCollapse"]["BA"],
                     "mean_random_projector_BA": one["random_BA"], "intact_WS_BA": one["Intact"]["WS_BA"], "ScaleCollapse_WS_BA": one["ScaleCollapse"]["WS_BA"],
                     "mean_random_projector_WS_BA": one["random_WS_BA"]})
    csv_write(RANK / "seed0_rankmatched_subject_metrics.csv", rows)
    summary = []
    for metric, i_col, s_col, r_col in (("future_S2_BA", "intact_BA", "ScaleCollapse_BA", "mean_random_projector_BA"), ("WS_BA", "intact_WS_BA", "ScaleCollapse_WS_BA", "mean_random_projector_WS_BA")):
        x = pd.DataFrame(rows); intact, scale, random = x[i_col].to_numpy(float), x[s_col].to_numpy(float), x[r_col].to_numpy(float)
        excess = 100 * (random - scale); mean, lo, hi = rank_summary.bootstrap(excess, "BNCI2015_001", metric)
        fixed_harm = float(np.mean(100 * (intact - scale)))
        random_harms = pd.DataFrame(projector_effects).query("metric == @metric").groupby("projector_id").random_harm_pp.mean().sort_index().to_numpy(float)
        percentile = float(100 * (np.mean(random_harms < fixed_harm) + .5 * np.mean(random_harms == fixed_harm)))
        summary.append({"metric": metric, "biological_subjects": 12, "intact": float(intact.mean()), "ScaleCollapse": float(scale.mean()), "random_rank16_mean": float(random.mean()),
                        "ScaleCollapse_harm_pp": fixed_harm, "random_rank16_harm_pp": float(random_harms.mean()), "excess_ScaleCollapse_harm_pp": mean,
                        "CI95_low_pp": lo, "CI95_high_pp": hi, "ScaleCollapse_percentile_among_random": percentile, "bootstrap_draws": 20000})
    csv_write(RANK / "seed0_rankmatched_summary.csv", summary); csv_write(RANK / "seed0_rankmatched_projector_effects.csv", projector_effects)
    manifest = pd.read_csv(RANK_PROTOCOL / "RANDOM_PROJECTOR_MANIFEST.csv").to_dict("records")
    json_write(RANK / "projector_manifest.json", {"reuse": "exact same pre-frozen bank for every BNCI fold", "source_audit": source["rank_projector_freeze"], "projectors": manifest,
               "scale_projector": "(J_3 / 3) tensor I_16", "insertion": "post-branch-dropout / pre-backend 48-channel tensor", "identity_wrapper_max_abs_logit_difference": identity_max})
    text_write(RANK / "RANKMATCHED_PROTOCOL.md", ["# BNCI rank-matched ScaleCollapse control", "",
        "This is frozen-checkpoint inference only. Identity is replayed before intervention. The fixed B2-like projector is (J3/3) tensor I16 at the exact post-branch-dropout/pre-backend 48-channel tensor; the same 100 frozen Gaussian-QR rank-16 projectors are reused in every fold.",
        "Random projectors are control conditions, not bootstrap units. All CIs use 20,000 paired biological-subject bootstrap draws."])
    text_write(RANK / "REPLAY_AUDIT.md", ["# BNCI frozen Full replay audit", "", f"All {len(replay)} frozen outer Full subject/session cells were recomputed before any intervention.",
        "Every BA and Macro-F1 cell matched the frozen result within 1e-7; intervention gate: PASS.", f"Identity-wrapper maximum absolute logit difference: `{identity_max:.9g}` (threshold `< 1e-5`): PASS."])
    lines = ["# BNCI2015-001 seed-0 rank-matched ScaleCollapse summary", "", "Positive excess harm means fixed scale collapse caused more loss than the mean same-rank random projection. Percentiles are descriptive only.", "",
             "| Metric | Intact | ScaleCollapse | Random rank-16 | Scale harm pp | Random harm pp | Excess harm pp [95% CI] | Scale percentile |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary: lines.append(f"| {row['metric']} | {row['intact']:.4f} | {row['ScaleCollapse']:.4f} | {row['random_rank16_mean']:.4f} | {row['ScaleCollapse_harm_pp']:+.3f} | {row['random_rank16_harm_pp']:+.3f} | {row['excess_ScaleCollapse_harm_pp']:+.3f} [{row['CI95_low_pp']:+.3f}, {row['CI95_high_pp']:+.3f}] | {row['ScaleCollapse_percentile_among_random']:.1f} |")
    text_write(RANK / "SEED0_RANKMATCHED_SUMMARY.md", lines)
    print("BNCI_EXTERNAL_RANKMATCHED_COMPLETE", flush=True)


def finalize() -> None:
    full = pd.read_csv(ROOT / "seed0_summary.csv"); diag = pd.read_csv(DIAG / "seed0_diagnostic_summary.csv"); arch = pd.read_csv(ARCH / "seed0_ablation_summary.csv"); rank = pd.read_csv(RANK / "seed0_rankmatched_summary.csv")
    models = full[full.row_type == "model"].set_index("model")
    lines = ["# BNCI2015-001 external replication — complete seed-0 synthesis", "", "This report is descriptive. The original Full SIRE-EEG and EEGNet seed-0 results remain frozen.", "",
             "## 1. Full-model external prediction", "", "| Model | Future S2 BA | Future S2 Macro-F1 | WS-BA |", "|---|---:|---:|---:|"]
    for model in ("EEGNet", "SIRE-EEG"):
        r = models.loc[model]; lines.append(f"| {model} | {r['future S2 BA']:.4f} | {r['future S2 Macro-F1']:.4f} | {r['WS-BA']:.4f} |")
    lines += ["", "### Paired SIRE-EEG minus EEGNet", "", "| Metric | Mean delta | 95% CI | Bootstrap unit |", "|---|---:|---:|---|"]
    for r in full[full.row_type == "paired_contrast"].itertuples(index=False):
        lines.append(f"| {r.metric} | {r.mean:+.4f} | [{r.CI95_low:+.4f}, {r.CI95_high:+.4f}] | biological subject, 20,000 draws |")
    lines += ["", "## 2. External PERSIST diagnostics", "", "| Diagnostic | Model | Mean pp [95% CI] | Coverage |", "|---|---|---:|---:|"]
    for r in diag.itertuples(index=False):
        effect = "undefined" if not pd.notna(r.mean_pp) else f"{r.mean_pp:.3f} [{r.CI95_low_pp:.3f}, {r.CI95_high_pp:.3f}]"
        lines.append(f"| {r.diagnostic} | {r.model} | {effect} | {r.nonempty_coverage} |")
    lines += ["", "PEEH/PSWA are diagnostic quantities and are not model-quality scores.", "", "## 3. External architecture replication", "", "| Model | Future S2 BA | WS-BA |", "|---|---:|---:|"]
    for r in arch[arch.row_type == "model"].itertuples(index=False): lines.append(f"| {r.model} | {r.future_S2_BA:.4f} | {r.WS_BA:.4f} |")
    lines += ["", "### Paired variant minus Full", "", "| Variant | Metric | Mean delta | 95% CI |", "|---|---|---:|---:|"]
    for r in arch[arch.row_type == "paired_contrast"].itertuples(index=False):
        lines.append(f"| {r.model} | {r.paired_metric} | {r.paired_delta_mean:+.4f} | [{r.CI95_low:+.4f}, {r.CI95_high:+.4f}] |")
    lines += ["", "No architecture was selected from these outcomes.", "", "## 4. Rank-matched ScaleCollapse control", "", "| Metric | Scale harm pp | Random rank-16 harm pp | Excess pp [95% CI] |", "|---|---:|---:|---:|"]
    for r in rank.itertuples(index=False): lines.append(f"| {r.metric} | {r.ScaleCollapse_harm_pp:+.3f} | {r.random_rank16_harm_pp:+.3f} | {r.excess_ScaleCollapse_harm_pp:+.3f} [{r.CI95_low_pp:+.3f}, {r.CI95_high_pp:+.3f}] |")
    text_write(ROOT / "BNCI2015_001_SEED0_COMPLETE_SUMMARY.md", lines)
    text_write(ROOT / "WHAT_REPLICATED_AND_WHAT_DID_NOT.md", ["# What replicated and what did not", "",
        "The output files report one BNCI2015-001 seed-0 matched-episodic external replication, frozen-checkpoint PERSIST diagnostics, pre-existing P3 architecture variants, and an inference-only fixed rank-matched ScaleCollapse control.",
        "They do not establish that PEEH or PSWA explains any prediction difference. PEEH/PSWA characterize selected subspaces; architecture ablations answer architecture questions; the rank-matched control compares a fixed scale-aligned intervention with equal-rank random removals.",
        "No seed 1/2, additional dataset, baseline, selector, adaptation, hyperparameter tuning, BN-state test, suppression, PRD, or other experiment was run."])
    print("BNCI_EXTERNAL_SEED0_FINALIZED", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=("diagnostics", "architecture", "rankmatched", "finalize"))
    args = parser.parse_args(); {"diagnostics": run_diagnostics, "architecture": run_architecture, "rankmatched": run_rankmatched, "finalize": finalize}[args.stage]()


if __name__ == "__main__":
    main()
