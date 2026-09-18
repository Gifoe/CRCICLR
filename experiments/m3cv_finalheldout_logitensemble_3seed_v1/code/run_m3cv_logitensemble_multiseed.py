"""M3CV final-heldout, three-seed, five-checkpoint logit-ensemble analysis.

This is deliberately a wrapper around the already-audited final M3CV runner.
It imports its final model classes and its training implementation directly.
Seed 0 checkpoints are immutable inputs.  Only seeds 1 and 2 are trained here.
No final-heldout array is opened until all 30 model checkpoints (5 folds x 3
seeds x 2 models) have been frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path("/root/p4_m3cv_finalheldout_logitensemble_3seed_v1")
BASE_ROOT = Path("/root/p4_m3cv_finalheldout_seed0_v1")
BASE_CODE = BASE_ROOT / "code/run_m3cv_finalheldout.py"
PEEH_CODE = Path("/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK/experiments/persist_eeg_litebn_ablation_v2_seed0/code/run_peeh_bridge.py")
PSWA_CODE = Path("/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK/experiments/persist_eeg_litebn_ablation_v2_seed0/code/run_pswa_bridge.py")
PEEH_SHA = "215cc8b393cc08f21f2536d9049527156e69230a08e5261a24abd8eb2c68dcc4"
PSWA_SHA = "7d04ff712b1eaaa4898b8ddff43b455c8e284dc706c6bd3af7a5a3a8e42c5251"
SEEDS = (0, 1, 2)
MODELS = ("EEGNet", "SIRE-EEG")
TASK = "M3CV_finalheldout_logitensemble"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def json_default(value: Any) -> Any:
    """Lossless JSON conversion for diagnostic artifacts with NumPy scalars."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader(); w.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def logits(model: torch.nn.Module, cell: Any, mean: np.ndarray, std: np.ndarray,
           base: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    x, y = base.open_cell(cell)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), base.BATCH_SIZE):
            z = np.asarray(x[start:start + base.BATCH_SIZE], dtype=np.float32).copy()
            z -= mean[None, :, None]
            z /= np.maximum(std[None, :, None], 1e-6)
            outputs.append(model(torch.from_numpy(z).to(device, non_blocking=True))[0].float().cpu().numpy())
    return np.concatenate(outputs), np.asarray(y, dtype=np.int64)


def embeddings(model: torch.nn.Module, cells: list[Any], mean: np.ndarray, std: np.ndarray,
               base: Any, device: torch.device, session_number: Mapping[str, int]) -> dict[str, Any]:
    hs: list[np.ndarray] = []; ys: list[np.ndarray] = []
    subs: list[np.ndarray] = []; sess: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for cell in cells:
            x, y = base.open_cell(cell)
            pieces: list[np.ndarray] = []
            for start in range(0, len(x), base.BATCH_SIZE):
                z = np.asarray(x[start:start + base.BATCH_SIZE], dtype=np.float32).copy()
                z -= mean[None, :, None]
                z /= np.maximum(std[None, :, None], 1e-6)
                _, h = model(torch.from_numpy(z).to(device, non_blocking=True))
                pieces.append(h.float().cpu().numpy())
            h = np.concatenate(pieces)
            if h.ndim != 2 or h.shape[1] != 64 or len(h) != len(y):
                raise RuntimeError("embedding schema drift")
            hs.append(h.astype(np.float32)); ys.append(np.asarray(y, dtype=np.int64))
            subs.append(np.repeat(cell.subject, len(y))); sess.append(np.repeat(session_number[cell.session], len(y)))
    return {"h": np.concatenate(hs), "y": np.concatenate(ys),
            "subject": np.concatenate(subs), "session": np.concatenate(sess)}


def load_normalizer(path: Path, expected_sha: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
        info = json.loads(str(z["info"]))
    got = hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()
    if got != expected_sha or info["mean_std_sha256"] != expected_sha:
        raise RuntimeError(f"normalizer provenance mismatch {path}")
    return mean, std, info


def records_from(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    result = value["records"]
    if len(result) != 10:
        raise RuntimeError(f"expected 10 model-fold records in {path}")
    for r in result:
        if not Path(r["checkpoint"]).is_file() or sha(Path(r["checkpoint"])) != r["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint provenance mismatch {r['checkpoint']}")
    return result


def get_seed_records(base: Any, cells: Mapping[Any, Any], development: list[str], seed: int,
                     sire: Any, eeg: Any, audit: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return frozen records and folds.  Existing complete seeds are immutable."""
    if seed == 0:
        manifest = json.loads((BASE_ROOT / "FINAL_HELDOUT_MANIFEST.json").read_text(encoding="utf-8"))
        folds = base.folds(development, 0)
        if manifest["folds"] != folds:
            raise RuntimeError("seed-0 fold manifest differs from authoritative source")
        return records_from(BASE_ROOT / "seed0_training_records.json"), folds
    records_path = ROOT / "development" / f"seed{seed}_training_records.json"
    folds = base.folds(development, seed)
    if records_path.is_file():
        return records_from(records_path), folds
    records: list[dict[str, Any]] = []
    for fold in folds:
        mean, std, info = base.normalize(cells, fold["train"])
        norm = ROOT / "normalizers" / f"seed{seed}_fold{fold['fold']}.npz"
        norm.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(norm, mean=mean, std=std, info=json.dumps(info, sort_keys=True))
        for name in MODELS:
            records.append(base.train_one(name, seed, fold, cells, mean, std, info, sire, eeg, audit, device))
    write_json(records_path, {"seed": seed, "records": records, "frozen_before_final_heldout_access": True})
    return records, folds


def normalizers_by_seed(base: Any, records: Mapping[int, list[dict[str, Any]]],
                        folds: Mapping[int, list[dict[str, Any]]], cells: Mapping[Any, Any]) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for seed, seed_folds in folds.items():
        for fold in seed_folds:
            expected = next(r["normalizer_sha256"] for r in records[seed] if r["fold"] == fold["fold"])
            # The old immutable seed-0 normalizers use the same exact schema.
            root = BASE_ROOT if seed == 0 else ROOT
            path = root / "normalizers" / f"seed{seed}_fold{fold['fold']}.npz"
            if not path.is_file():
                raise RuntimeError(f"missing frozen normalizer {path}")
            out[(seed, fold["fold"])] = load_normalizer(path, expected)
    return out


def ensemble_metrics(base: Any, sire: Any, eeg: Any, records: Mapping[int, list[dict[str, Any]]],
                     norms: Mapping[tuple[int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]],
                     heldout: list[str], cells: Mapping[Any, Any], device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for name in MODELS:
            recs = sorted((r for r in records[seed] if r["model"] == name), key=lambda r: r["fold"])
            if len(recs) != 5:
                raise RuntimeError("logit ensemble needs exactly five frozen checkpoints")
            for subject in heldout:
                for session in base.SESSIONS:
                    all_logits: list[np.ndarray] = []; label: np.ndarray | None = None
                    for r in recs:
                        mean, std, _ = norms[(seed, r["fold"])]
                        model = base.checkpoint_model(r, sire, eeg, device)
                        q, y = logits(model, cells[(subject, session)], mean, std, base, device)
                        del model
                        if label is None: label = y
                        elif not np.array_equal(label, y): raise RuntimeError("label drift across ensemble members")
                        all_logits.append(q)
                    if device.type == "cuda": torch.cuda.empty_cache()
                    average = np.stack(all_logits, axis=0).mean(axis=0)
                    prediction = average.argmax(1)
                    rows.append({"model": name, "seed": seed, "subject": subject, "session": session,
                                 "checkpoints_logit_averaged": 5, "trials": int(len(label)),
                                 "BA": float(balanced_accuracy_score(label, prediction)),
                                 "Macro_F1": float(f1_score(label, prediction, average="macro", zero_division=0))})
    return rows


def aggregate_performance(base: Any, ensemble: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        for name in MODELS:
            subjects = sorted({r["subject"] for r in ensemble if r["seed"] == seed and r["model"] == name}, key=base.subject_key)
            vals = []
            for subject in subjects:
                x = {r["session"]: r for r in ensemble if r["seed"] == seed and r["model"] == name and r["subject"] == subject}
                if set(x) != set(base.SESSIONS): raise RuntimeError("subject session ensemble coverage drift")
                vals.append((x["ses-02"]["BA"], x["ses-02"]["Macro_F1"], min(x[s]["BA"] for s in base.SESSIONS)))
            per_seed.append({"row_type": "model", "seed": seed, "model": name, "subjects": len(subjects),
                             "S2_BA": float(np.mean([x[0] for x in vals])), "S2_Macro_F1": float(np.mean([x[1] for x in vals])), "WS_BA": float(np.mean([x[2] for x in vals]))})
    subject_rows: list[dict[str, Any]] = []
    for name in MODELS:
        subjects = sorted({r["subject"] for r in ensemble if r["model"] == name}, key=base.subject_key)
        for subject in subjects:
            one = []
            for seed in SEEDS:
                x = {r["session"]: r for r in ensemble if r["model"] == name and r["subject"] == subject and r["seed"] == seed}
                if set(x) != set(base.SESSIONS): raise RuntimeError("three-seed session coverage drift")
                one.append((x["ses-02"]["BA"], x["ses-02"]["Macro_F1"], min(x[s]["BA"] for s in base.SESSIONS)))
            subject_rows.append({"model": name, "subject": subject, "seeds_averaged": 3,
                                 "S2_BA": float(np.mean([x[0] for x in one])), "S2_Macro_F1": float(np.mean([x[1] for x in one])), "WS_BA": float(np.mean([x[2] for x in one]))})
    summary: list[dict[str, Any]] = []
    for name in MODELS:
        x = [r for r in subject_rows if r["model"] == name]
        summary.append({"row_type": "model", "model": name, "subjects": len(x),
                        "S2_BA": float(np.mean([r["S2_BA"] for r in x])), "S2_Macro_F1": float(np.mean([r["S2_Macro_F1"] for r in x])), "WS_BA": float(np.mean([r["WS_BA"] for r in x])),
                        "contrast": "", "mean": "", "CI95_low": "", "CI95_high": ""})
    a = {r["subject"]: r for r in subject_rows if r["model"] == "SIRE-EEG"}; b = {r["subject"]: r for r in subject_rows if r["model"] == "EEGNet"}
    for metric in ("S2_BA", "WS_BA"):
        v = np.asarray([a[s][metric] - b[s][metric] for s in sorted(a, key=base.subject_key)], np.float64)
        rng = np.random.default_rng(stable_seed("M3CV-logitensemble-bootstrap", metric))
        draw = v[rng.integers(0, len(v), size=(20_000, len(v)))].mean(1)
        summary.append({"row_type": "paired_SIRE_minus_EEGNet", "model": "SIRE-EEG minus EEGNet", "subjects": len(v),
                        "S2_BA": "", "S2_Macro_F1": "", "WS_BA": "", "contrast": metric, "mean": float(v.mean()), "CI95_low": float(np.quantile(draw, .025)), "CI95_high": float(np.quantile(draw, .975))})
    return per_seed, subject_rows, summary


def peeh_empty(result: dict[str, Any]) -> dict[str, Any]:
    if result["protected_rank"] != 0:
        return result
    for row in result["subject_rows"]:
        intact = float(row["intact_probe_BA"])
        row.update({"protected_erased_BA": intact, "random_erased_BA": intact,
                    "protected_harm_pp": 0.0, "random_harm_pp": 0.0, "PEEH_pp": 0.0})
    result.update({"protected_erased_BA": result["intact_probe_BA"], "random_erased_BA": result["intact_probe_BA"],
                   "protected_harm_pp": 0.0, "random_harm_pp": 0.0, "PEEH_pp": 0.0,
                   "empty_protected_effect_rule": "exactly_zero"})
    return result


def pswa_one(pswa: Any, peeh: Mapping[str, Any], train: Mapping[str, Any], ev: Mapping[str, Any],
             base: Any, model: str, fold: int, seed: int) -> dict[str, Any]:
    rank = int(peeh["protected_rank"]); active = int(peeh["active_rank"])
    common = {"task": TASK, "model": model, "fold": fold, "seed": seed, "protected_rank": rank, "active_rank": active,
              "protected_coordinates": list(map(int, peeh["protected_coordinates"]))}
    if rank == 0:
        return {**common, "status": "EMPTY_PROTECTED_UNDEFINED_OMITTED", "subject_rows": [], "random_draws": 0}
    basis = pswa.canonical_basis(train["h"], train["subject"], train["session"], train["y"], peeh)
    ztrain = pswa.z_coordinates(train["h"], basis); zeval = pswa.z_coordinates(ev["h"], basis)
    protected = common["protected_coordinates"]
    ppack = pswa.ridge_fit(ztrain[:, protected], train["y"], 2)
    ppred = pswa.ridge_predict(zeval[:, protected], ppack)
    random = pswa.random_sets(TASK, model, fold, seed, rank, active)
    if len(random) != 100:
        raise RuntimeError("frozen PSWA must use 100 equal-rank controls")
    subjects = sorted(np.unique(ev["subject"]).astype(str), key=base.subject_key)
    pba: dict[tuple[str, int], float] = {}
    rba: dict[tuple[int, str, int], float] = {}
    for subject in subjects:
        for session in (1, 2):
            mask = (ev["subject"].astype(str) == subject) & (ev["session"].astype(int) == session)
            if not mask.any() or set(np.unique(ev["y"][mask]).tolist()) != {0, 1}: raise RuntimeError("invalid PSWA evaluation cell")
            pba[(subject, session)] = pswa.ba(ev["y"][mask], ppred[mask], 2)
    for draw, subset in enumerate(random):
        pack = pswa.ridge_fit(ztrain[:, subset], train["y"], 2); pred = pswa.ridge_predict(zeval[:, subset], pack)
        for subject in subjects:
            for session in (1, 2):
                mask = (ev["subject"].astype(str) == subject) & (ev["session"].astype(int) == session)
                rba[(draw, subject, session)] = pswa.ba(ev["y"][mask], pred[mask], 2)
    rows = []
    for subject in subjects:
        pws = min(pba[(subject, 1)], pba[(subject, 2)])
        rws = np.asarray([min(rba[(d, subject, 1)], rba[(d, subject, 2)]) for d in range(100)], np.float64)
        rows.append({**common, "status": "VALID", "subject": subject, "protected_S1_BA": pba[(subject, 1)], "protected_S2_BA": pba[(subject, 2)],
                     "random_S1_BA": float(np.mean([rba[(d, subject, 1)] for d in range(100)])), "random_S2_BA": float(np.mean([rba[(d, subject, 2)] for d in range(100)])),
                     "protected_only_WSBA": pws, "random_only_WSBA": float(rws.mean()), "PSWA_pp": float(100 * (pws - rws.mean())), "random_draws": 100})
    return {**common, "status": "VALID", "subject_rows": rows, "random_draws": 100,
            "basis_active_rank": int(len(basis["rho"]))}


def diagnostics(base: Any, sire: Any, eeg: Any, records: Mapping[int, list[dict[str, Any]]], folds: Mapping[int, list[dict[str, Any]]],
                norms: Mapping[tuple[int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]], heldout: list[str], cells: Mapping[Any, Any], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    peeh = load_module("m3cv_frozen_peeh", PEEH_CODE); pswa = load_module("m3cv_frozen_pswa", PSWA_CODE)
    peeh_runs = []; pswa_runs = []
    session_number = {"ses-01": 1, "ses-02": 2}
    held_s2 = [cells[(s, "ses-02")] for s in heldout]
    held_both = [cells[(s, q)] for s in heldout for q in base.SESSIONS]
    for seed in SEEDS:
        for fold in folds[seed]:
            mean, std, _ = norms[(seed, fold["fold"])]
            train_cells = [cells[(s, q)] for s in fold["train"] for q in base.SESSIONS]
            for name in MODELS:
                partial = ROOT / "diagnostics" / f"seed{seed}_fold{fold['fold']}_{name}.json"
                if partial.is_file():
                    saved = json.loads(partial.read_text(encoding="utf-8"))
                    peeh_runs.append(saved["peeh"])
                    pswa_runs.append(saved["pswa"])
                    print(f"DIAGNOSTIC_RESUME seed={seed} fold={fold['fold']} model={name}", flush=True)
                    continue
                r = next(x for x in records[seed] if x["model"] == name and x["fold"] == fold["fold"])
                model = base.checkpoint_model(r, sire, eeg, device)
                tr = embeddings(model, train_cells, mean, std, base, device, session_number)
                ev2 = embeddings(model, held_s2, mean, std, base, device, session_number)
                evb = embeddings(model, held_both, mean, std, base, device, session_number)
                del model
                if device.type == "cuda": torch.cuda.empty_cache()
                payload = {"train_h": tr["h"], "train_y": tr["y"], "train_subject": tr["subject"], "train_session": tr["session"],
                           "eval_h": ev2["h"], "eval_y": ev2["y"], "eval_subject": ev2["subject"], "eval_session": ev2["session"],
                           "metadata": {"checkpoint_sha256": r["checkpoint_sha256"], "normalizer_sha256": r["normalizer_sha256"], "representation": "64-d immediately before final classifier"}}
                result = peeh_empty(peeh.run_one(payload, TASK, name, fold["fold"], seed, 2))
                peeh_runs.append(result)
                pswa_result = pswa_one(pswa, result, tr, evb, base, name, fold["fold"], seed)
                pswa_runs.append(pswa_result)
                write_json(partial, {"peeh": result, "pswa": pswa_result})
                print(f"DIAGNOSTIC seed={seed} fold={fold['fold']} model={name} protected_rank={result['protected_rank']}", flush=True)
    peeh_subject = []
    for name in MODELS:
        for subject in heldout:
            selected = []
            ranks = []
            for result in peeh_runs:
                if result["model"] != name: continue
                ranks.append(int(result["protected_rank"]))
                row = next(x for x in result["subject_rows"] if x["subject_id"] == subject)
                selected.append(float(row["PEEH_pp"]))
            if len(selected) != 15: raise RuntimeError("PEEH run coverage drift")
            peeh_subject.append({"model": name, "subject": subject, "fold_seed_runs": 15, "PEEH_pp": float(np.mean(selected)), "protected_nonempty_runs": int(sum(x > 0 for x in ranks))})
    pswa_subject = []
    for name in MODELS:
        for subject in heldout:
            selected = [row for r in pswa_runs if r["model"] == name for row in r["subject_rows"] if row["subject"] == subject]
            if selected:
                pswa_subject.append({"model": name, "subject": subject, "PSWA_pp": float(np.mean([r["PSWA_pp"] for r in selected])),
                                     "protected_only_WSBA": float(np.mean([r["protected_only_WSBA"] for r in selected])), "random_only_WSBA": float(np.mean([r["random_only_WSBA"] for r in selected])),
                                     "nonempty_fold_seed_runs": len(selected), "total_fold_seed_runs": 15})
            else:
                pswa_subject.append({"model": name, "subject": subject, "PSWA_pp": "", "protected_only_WSBA": "", "random_only_WSBA": "", "nonempty_fold_seed_runs": 0, "total_fold_seed_runs": 15})
    summary = []
    for name in MODELS:
        pr = [r for r in peeh_runs if r["model"] == name]; ps = [r for r in pswa_runs if r["model"] == name]
        pv = np.asarray([r["PEEH_pp"] for r in peeh_subject if r["model"] == name], np.float64)
        rng = np.random.default_rng(stable_seed("M3CV-PEEH-bootstrap", name)); pd = pv[rng.integers(0, len(pv), size=(20_000, len(pv)))].mean(1)
        valid = [r for r in pswa_subject if r["model"] == name and r["PSWA_pp"] != ""]
        row = {"model": name, "PEEH_mean_pp": float(pv.mean()), "PEEH_CI95_low_pp": float(np.quantile(pd, .025)), "PEEH_CI95_high_pp": float(np.quantile(pd, .975)),
               "PEEH_nonempty_coverage": f"{sum(r['protected_rank'] > 0 for r in pr)}/15", "PEEH_empty_runs_zero_effect": int(sum(r['protected_rank'] == 0 for r in pr)),
               "PSWA_nonempty_coverage": f"{sum(r['protected_rank'] > 0 for r in ps)}/15", "PSWA_biological_subjects": len(valid), "PSWA_mean_pp": "", "PSWA_CI95_low_pp": "", "PSWA_CI95_high_pp": ""}
        if valid:
            v = np.asarray([r["PSWA_pp"] for r in valid], np.float64); rng = np.random.default_rng(stable_seed("M3CV-PSWA-bootstrap", name)); d = v[rng.integers(0, len(v), size=(20_000, len(v)))].mean(1)
            row.update({"PSWA_mean_pp": float(v.mean()), "PSWA_CI95_low_pp": float(np.quantile(d, .025)), "PSWA_CI95_high_pp": float(np.quantile(d, .975))})
        summary.append(row)
    write_json(ROOT / "diagnostics" / "run_level_peeh.json", peeh_runs)
    write_json(ROOT / "diagnostics" / "run_level_pswa.json", pswa_runs)
    return peeh_subject, pswa_subject, summary


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--skip-diagnostics", action="store_true"); args = p.parse_args()
    if not BASE_CODE.is_file(): raise RuntimeError("authoritative M3CV runner missing")
    if sha(PEEH_CODE) != PEEH_SHA or sha(PSWA_CODE) != PSWA_SHA: raise RuntimeError("frozen diagnostic source hash drift")
    base = load_module("m3cv_authoritative_base", BASE_CODE); base.ROOT = ROOT
    ROOT.mkdir(parents=True, exist_ok=True)
    sire, eeg, audit = base.load_models(); subjects, cells, order = base.inventory(); development, heldout = base.heldout_split(subjects)
    records: dict[int, list[dict[str, Any]]] = {}; folds: dict[int, list[dict[str, Any]]] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # This loop performs no heldout array access. Its output is the explicit freeze barrier.
    for seed in SEEDS:
        records[seed], folds[seed] = get_seed_records(base, cells, development, seed, sire, eeg, audit, device)
    norms = normalizers_by_seed(base, records, folds, cells)
    frozen = [r for seed in SEEDS for r in records[seed]]
    if len(frozen) != 30 or len({(r['seed'], r['model'], r['fold']) for r in frozen}) != 30: raise RuntimeError("checkpoint freeze coverage failure")
    write_json(ROOT / "CHECKPOINT_FREEZE_AUDIT.json", {"seeds": list(SEEDS), "models": list(MODELS), "checkpoints": [{k:r[k] for k in ("seed","model","fold","selected_epoch","checkpoint","checkpoint_sha256","normalizer_sha256")} for r in frozen],
               "all_30_model_checkpoints_frozen_before_new_finalheldout_array_access": True, "final_heldout_subjects": heldout})
    write_json(ROOT / "MANIFEST.json", {"dataset": "M3CV / NEMAR nm000166", "development_subjects": development, "final_heldout_subjects": heldout, "seed_folds": folds, "channel_order": list(order), "logit_ensemble": "arithmetic mean of raw two-class logits over exactly five folds within each seed", "three_seed_aggregation": "subject-level mean of seed-level five-checkpoint ensembles"})
    ensemble = ensemble_metrics(base, sire, eeg, records, norms, heldout, cells, device)
    write_csv(ROOT / "three_seed_logitensemble_checkpoint_ensemble_metrics.csv", ensemble, ["model","seed","subject","session","checkpoints_logit_averaged","trials","BA","Macro_F1"])
    per_seed, subject_metrics, summary = aggregate_performance(base, ensemble)
    write_csv(ROOT / "three_seed_logitensemble_per_seed_summary.csv", per_seed, ["row_type","seed","model","subjects","S2_BA","S2_Macro_F1","WS_BA"])
    write_csv(ROOT / "three_seed_logitensemble_subject_metrics.csv", subject_metrics, ["model","subject","seeds_averaged","S2_BA","S2_Macro_F1","WS_BA"])
    write_csv(ROOT / "three_seed_logitensemble_summary.csv", summary, ["row_type","model","subjects","S2_BA","S2_Macro_F1","WS_BA","contrast","mean","CI95_low","CI95_high"])
    diag_summary = []
    if not args.skip_diagnostics:
        peeh_subject, pswa_subject, diag_summary = diagnostics(base, sire, eeg, records, folds, norms, heldout, cells, device)
        write_csv(ROOT / "three_seed_peeh_subject_results.csv", peeh_subject, ["model","subject","fold_seed_runs","PEEH_pp","protected_nonempty_runs"])
        write_csv(ROOT / "three_seed_pswa_subject_results.csv", pswa_subject, ["model","subject","PSWA_pp","protected_only_WSBA","random_only_WSBA","nonempty_fold_seed_runs","total_fold_seed_runs"])
        write_csv(ROOT / "three_seed_diagnostic_summary.csv", diag_summary, ["model","PEEH_mean_pp","PEEH_CI95_low_pp","PEEH_CI95_high_pp","PEEH_nonempty_coverage","PEEH_empty_runs_zero_effect","PSWA_nonempty_coverage","PSWA_biological_subjects","PSWA_mean_pp","PSWA_CI95_low_pp","PSWA_CI95_high_pp"])
    by = {(r["row_type"], r["model"], r["contrast"]): r for r in summary}
    lines = ["# M3CV final-heldout three-seed logit ensemble", "", "Each result first averages raw logits over five frozen development-fold checkpoints for the same seed; it then averages the three seed-level values within each biological heldout subject.", "", "| Model | S2 BA | S2 Macro-F1 | WS-BA |", "|---|---:|---:|---:|"]
    for name in MODELS:
        r = by[("model", name, "")]; lines.append(f"| {name} | {100*float(r['S2_BA']):.2f}% | {100*float(r['S2_Macro_F1']):.2f}% | {100*float(r['WS_BA']):.2f}% |")
    lines += ["", "| SIRE-EEG minus EEGNet | Mean | 95% subject bootstrap CI |", "|---|---:|---:|"]
    for metric in ("S2_BA", "WS_BA"):
        r = by[("paired_SIRE_minus_EEGNet", "SIRE-EEG minus EEGNet", metric)]; lines.append(f"| {metric} | {100*float(r['mean']):+.2f} pp | [{100*float(r['CI95_low']):+.2f}, {100*float(r['CI95_high']):+.2f}] pp |")
    if diag_summary:
        lines += ["", "| Model | PEEH | PEEH coverage | PSWA | PSWA coverage |", "|---|---:|---:|---:|---:|"]
        for r in diag_summary:
            ps = "undefined/omitted" if r["PSWA_mean_pp"] == "" else f"{float(r['PSWA_mean_pp']):+.2f} pp [{float(r['PSWA_CI95_low_pp']):+.2f}, {float(r['PSWA_CI95_high_pp']):+.2f}]"
            lines.append(f"| {r['model']} | {float(r['PEEH_mean_pp']):+.2f} pp [{float(r['PEEH_CI95_low_pp']):+.2f}, {float(r['PEEH_CI95_high_pp']):+.2f}] | {r['PEEH_nonempty_coverage']} | {ps} | {r['PSWA_nonempty_coverage']} |")
    lines += ["", "PEEH empty Protected assignments are an exactly-zero effect. PSWA empty Protected assignments are undefined and omitted; coverage is nonempty fold/checkpoint runs out of 15, never a subject count."]
    write_text(ROOT / "THREE_SEED_LOGITENSEMBLE_SUMMARY.md", "\n".join(lines))
    print("M3CV_THREE_SEED_LOGITENSEMBLE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as e:
        print(f"M3CV_THREE_SEED_LOGITENSEMBLE_INVALID: {type(e).__name__}: {e}", flush=True)
        raise
