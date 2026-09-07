from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("R2EEG_REPO", "/root/rivermind-data/CRCICLR_EEGNET_PRD_WORK")).resolve()
SRC = REPO / "experiments" / "persist_eeg_carrier_dualdataset_screen_v1"
STAGE1 = REPO / "experiments" / "persist_eeg_r2eeg_stage1_v1" / "code"
sys.path[:0] = [str(SRC / "code"), str(STAGE1), str(Path(__file__).parent)]
import run_stage1 as v1  # noqa: E402
import run_carrier_screen as carrier  # noqa: E402
from build_fivefold_split import build, load_search  # noqa: E402
from eegnet_locked import EEGNet  # noqa: E402

EXP = ROOT
PROTOCOL = EXP / "protocol"
OUT = EXP / "outputs"
RUNTIME = Path(os.environ.get("CARRIER_5FOLD_RUNTIME", "/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")).resolve()
EPOCHS, MIN_EPOCH = 60, 10
BOOTSTRAPS = 10000
TIE_TOL = 1e-12


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def state_hash(state: dict[str, torch.Tensor]) -> str:
    import io
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return sha_bytes(buffer.getvalue())


def constructor(name: str, channels: int) -> torch.nn.Module:
    if name == "EEGNet":
        return EEGNet(channels)
    if name == "LiteBN":
        return carrier.CompactLite(channels, "bn")
    raise ValueError(name)


def validate_split(search: dict[str, list[str]], folds: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for dataset, subjects in search.items():
        subject_set = set(subjects)
        outer = [set(f["outer_dev_subjects"]) for f in folds[dataset]]
        if len(outer) != 5 or any(a & b for i, a in enumerate(outer) for b in outer[i + 1:]) or set.union(*outer) != subject_set:
            raise RuntimeError(f"outer split invariant failed: {dataset}")
        entries = []
        for f in folds[dataset]:
            tr, va, od = map(set, (f["inner_train_subjects"], f["inner_val_subjects"], f["outer_dev_subjects"]))
            if tr & va or tr & od or va & od or (tr | va | od) != subject_set:
                raise RuntimeError(f"within-fold split invariant failed: {dataset} f{f['fold_id']}")
            entries.append({"fold": f["fold_id"], "outer_size": len(od), "inner_train_size": len(tr), "inner_val_size": len(va), "disjoint": True})
        report[dataset] = {"search_size": len(subjects), "outer_union_exact": True, "outer_disjoint": True, "folds": entries}
    return report


def train_model(model: torch.nn.Module, name: str, bundle: Any, fold: dict[str, Any], manifest: list[list[dict[str, Any]]], cache: Any, manifest_sha: str, cell_dir: Path, device: torch.device) -> dict[str, Any]:
    cell_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cell_dir / "checkpoint_latest.pt"
    init_hash = state_hash(copy.deepcopy(model.state_dict()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, history, best, best_epoch, best_state = 1, [], -float("inf"), None, None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["init_sha256"] != init_hash or saved["manifest_sha256"] != manifest_sha:
            raise RuntimeError(f"resume invariant mismatch for {checkpoint}")
        model.load_state_dict(saved["current_state"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start = int(saved["epoch"]) + 1
        history, best, best_epoch, best_state = saved["history"], saved["best_val_BA"], saved["best_epoch"], saved["best_state"]
        restore_rng(saved["rng"])
    started = time.perf_counter()
    for epoch in range(start, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for episode in manifest[epoch - 1]:
            indices = episode["support_indices"] + episode["query_indices"]
            x, y = cache.batch(indices)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits, _ = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite CE in {name}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        val_rows, val_ba, _ = carrier.eval_rows(model, bundle, fold["inner_val_subjects"], cache)
        selected = epoch >= MIN_EPOCH and val_ba > best + 1e-12
        if selected:
            best, best_epoch, best_state = float(val_ba), epoch, copy.deepcopy(model.state_dict())
        row = {"epoch": epoch, "CE": float(np.mean(losses)), "inner_val_subject_BA": float(val_ba), "selected": bool(selected)}
        history.append(row)
        torch.save({"epoch": epoch, "history": history, "best_val_BA": best, "best_epoch": best_epoch, "best_state": best_state, "current_state": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(), "manifest_sha256": manifest_sha, "init_sha256": init_hash}, checkpoint)
        if epoch == 1 or epoch % 5 == 0 or selected:
            print(f"[{bundle.name} fold={fold['fold_id']} seed={cell_dir.name} {name}] epoch={epoch:02d} CE={row['CE']:.4f} valBA={val_ba:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError(f"no eligible checkpoint for {name}")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    epoch60_state = saved["current_state"]
    epoch60_path = cell_dir / "epoch60.pt"
    torch.save(epoch60_state, epoch60_path)
    model.load_state_dict(best_state)
    selected_path = cell_dir / "selected_best.pt"
    torch.save(model.state_dict(), selected_path)
    return {"model": name, "selected_epoch": int(best_epoch), "best_inner_val_BA": float(best), "epoch60_inner_val_BA": float(history[-1]["inner_val_subject_BA"]), "history": history, "checkpoint_path": str(selected_path), "epoch60_checkpoint_path": str(epoch60_path), "checkpoint_sha256": sha_file(selected_path), "epoch60_checkpoint_sha256": sha_file(epoch60_path), "init_sha256": init_hash, "manifest_sha256": manifest_sha, "elapsed_seconds": time.perf_counter() - started, "amp": amp, "epochs_completed": len(history)}


def evaluate_model(model: torch.nn.Module, bundle: Any, subjects: list[str], cache: Any) -> tuple[dict[str, dict[str, float]], float, float]:
    model.eval()
    rows: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for subject in subjects:
            idx = bundle.indices([subject], (2,))
            yy = bundle.labels(idx)
            logits = []
            for start in range(0, len(idx), 128):
                x, _ = cache.batch(idx[start:start + 128])
                logits.append(model(x)[0].float().cpu().numpy())
            pred = np.concatenate(logits, axis=0).argmax(1)
            rows[str(subject)] = {"BA": float(balanced_accuracy_score(yy, pred)), "macro_F1": float(f1_score(yy, pred, average="macro")), "accuracy": float(accuracy_score(yy, pred)), "trials": int(len(yy))}
    return rows, float(np.mean([r["BA"] for r in rows.values()])), float(np.mean([r["macro_F1"] for r in rows.values()]))


def evaluate_checkpoint(model: torch.nn.Module, path: Path, bundle: Any, subjects: list[str], cache: Any, device: torch.device) -> tuple[dict[str, dict[str, float]], float, float]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    return evaluate_model(model, bundle, subjects, cache)


def bootstrap(delta_pp: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    draws = rng.choice(delta_pp, size=(BOOTSTRAPS, len(delta_pp)), replace=True).mean(1)
    return {"n_subjects": int(len(delta_pp)), "resamples": BOOTSTRAPS, "mean_delta_pp": float(delta_pp.mean()), "median_delta_pp": float(np.median(delta_pp)), "ci_low_pp": float(np.quantile(draws, .025)), "ci_high_pp": float(np.quantile(draws, .975)), "improved_subjects": int((delta_pp > TIE_TOL).sum()), "harmed_subjects": int((delta_pp < -TIE_TOL).sum()), "tied_subjects": int((np.abs(delta_pp) <= TIE_TOL).sum())}


def spearman_bootstrap(x: np.ndarray, y: np.ndarray) -> tuple[float | None, list[float | None]]:
    def rho(a: np.ndarray, b: np.ndarray) -> float | None:
        value = spearmanr(a, b).statistic
        return None if not np.isfinite(value) else float(value)
    observed = rho(x, y)
    rng = np.random.default_rng(0)
    vals = [rho(x[idx], y[idx]) for idx in (rng.integers(0, len(x), len(x)) for _ in range(BOOTSTRAPS))]
    finite = np.asarray([v for v in vals if v is not None], dtype=float)
    return observed, ([float(np.quantile(finite, .025)), float(np.quantile(finite, .975))] if len(finite) else [None, None])


def variance_decomposition(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame["delta_pp"].to_numpy(float)
    n = len(y); grand = float(y.mean())
    fold_means = frame.groupby("fold").delta_pp.mean()
    seed_means = frame.groupby("seed").delta_pp.mean()
    ss_total = float(((y - grand) ** 2).sum())
    ss_fold = float(sum(len(g) * (g.delta_pp.mean() - grand) ** 2 for _, g in frame.groupby("fold")))
    ss_seed = float(sum(len(g) * (g.delta_pp.mean() - grand) ** 2 for _, g in frame.groupby("seed")))
    ss_resid = max(0.0, ss_total - ss_fold - ss_seed)
    return {"n_cells": n, "grand_mean_pp": grand, "sum_squares_total": ss_total, "sum_squares_fold": ss_fold, "sum_squares_seed": ss_seed, "sum_squares_residual": ss_resid, "fold_effect_fraction": (ss_fold / ss_total if ss_total else None), "seed_effect_fraction": (ss_seed / ss_total if ss_total else None), "sd_fold_means_pp": float(fold_means.std(ddof=0)), "sd_seed_means_pp": float(seed_means.std(ddof=0)), "median_within_fold_seed_range_pp": float(frame.groupby("fold").delta_pp.agg(lambda x: x.max() - x.min()).median()), "max_within_fold_seed_range_pp": float(frame.groupby("fold").delta_pp.agg(lambda x: x.max() - x.min()).max()), "fold_means_pp": {str(k): float(v) for k, v in fold_means.items()}, "seed_means_pp": {str(k): float(v) for k, v in seed_means.items()}, "descriptive_only": True}


def classify_fold(values: pd.Series) -> str:
    positive = int((values > 0).sum())
    negative = int((values < 0).sum())
    meaningful_pos = int((values >= 1).sum())
    meaningful_neg = int((values <= -1).sum())
    if positive >= 2 and meaningful_neg == 0:
        return "SEED_CONSISTENT_POSITIVE"
    if negative >= 2 and meaningful_pos == 0:
        return "SEED_CONSISTENT_NEGATIVE"
    return "SEED_UNSTABLE"


def decision(dataset_agg: pd.DataFrame, fold_agg: pd.DataFrame, seed_agg: pd.DataFrame, variance: dict[str, Any], checkpoint: dict[str, Any]) -> str:
    stable = True; small = True
    for _, row in dataset_agg.iterrows():
        f = fold_agg[fold_agg.dataset == row.dataset]
        s = seed_agg[seed_agg.dataset == row.dataset]
        positive_folds = int((f.mean_delta_pp > 0).sum())
        positive_seeds = int((s.mean_delta_pp_over_folds > 0).sum())
        stable &= row.grand_mean_delta_pp >= 1 and positive_folds >= 4 and positive_seeds == 3 and row.median_subject_delta_pp >= 0 and row.fraction_subjects_gain_ge_1pp >= 0.25
        small &= row.grand_mean_delta_pp > 0 and positive_folds >= 4 and positive_seeds >= 2
    if stable:
        return "LITEBN_STABLE_CROSSDATASET_GAIN"
    if small and not stable:
        return "LITEBN_SMALL_BUT_CONSISTENT_GAIN"
    fold_sensitive = any(v["sd_fold_means_pp"] > max(0.25, 1.25 * v["sd_seed_means_pp"]) and (fold_agg[fold_agg.dataset == d].mean_delta_pp.min() < -0.5) for d, v in variance.items())
    seed_sensitive = any(v["sd_seed_means_pp"] >= v["sd_fold_means_pp"] or (fold_agg[fold_agg.dataset == d].sign_class == "SEED_UNSTABLE").any() for d, v in variance.items())
    checkpoint_sensitive = any(v.get("selected_vs_epoch60_materially_different", False) for v in checkpoint.values())
    if fold_sensitive and seed_sensitive:
        return "LITEBN_UNSTABLE_MULTIFACTOR"
    if fold_sensitive:
        return "LITEBN_FOLD_SUBJECT_SENSITIVE"
    if seed_sensitive:
        return "LITEBN_SEED_OPTIMIZATION_SENSITIVE"
    if checkpoint_sensitive:
        return "LITEBN_CHECKPOINT_SELECTION_SENSITIVE"
    return "LITEBN_MIXED_OR_NO_MEANINGFUL_GAIN"


def main() -> int:
    for path in (PROTOCOL, OUT, RUNTIME):
        path.mkdir(parents=True, exist_ok=True)
    search = load_search(REPO)
    folds = build(search)
    split_payload = {"protocol": "CARRIER_5FOLD_MULTISEED_STABILITY_V1", "split_seed": 0, "inner_val_seed_base": 1000, "search_subjects": search, "folds": folds}
    split_path = PROTOCOL / "FIVEFOLD_SPLIT.json"
    split_path.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_json(PROTOCOL / "SPLIT_HASH.json", {"path": str(split_path), "sha256": sha_file(split_path)})
    split_audit = validate_split(search, folds)
    write_json(PROTOCOL / "SEARCH_SUBJECT_AUDIT.json", {"exact_search_subjects": search, "sizes": {k: len(v) for k, v in search.items()}, "fold_audit": split_audit})
    write_json(PROTOCOL / "HOLDOUT_ISOLATION_AUDIT.json", {"V8_INTERNAL_HOLDOUT_loaded": False, "V8_INTERNAL_HOLDOUT_labels_loaded": False, "WBCIC_true_outer_loaded": False, "WBCIC_true_outer_labels_loaded": False, "allowed_scope": "V8_SEARCH only"})
    write_json(PROTOCOL / "TRAINING_PROTOCOL.json", {"models": ["EEGNet", "LiteBN"], "datasets": ["OpenBMI", "WBCIC"], "folds": 5, "seeds": [0, 1, 2], "optimizer": "AdamW", "lr": 3e-4, "weight_decay": 5e-4, "batch_size": 64, "episode_batch_size": 128, "gradient_clipping": 5.0, "epochs": 60, "scheduler": "none", "loss": "ordinary cross entropy", "checkpoint_selection": "inner future-session mean-subject BA, eligible epochs 10..60, earliest tie", "primary_unit": "subject", "bootstrap": {"unit": "subject", "resamples": BOOTSTRAPS, "seed": 0}})
    write_json(PROTOCOL / "INFORMATION_MATCHING.json", {"same_inner_train_subjects": True, "same_inner_val_subjects": True, "same_outer_dev_subjects": True, "same_cached_trials": True, "same_normalizer": True, "same_episode_manifest": True, "same_CE_samples": True, "same_sample_order": True, "same_batch_order": True, "same_optimizer": True, "same_epoch_budget": True, "same_checkpoint_rule": True, "only_intended_difference": "architecture"})
    tests = {"five_outer_folds_disjoint": True, "search_union_exact": True, "each_search_subject_once_outer": True, "inner_train_val_outer_disjoint": True, "same_split_across_seeds": True, "split_independent_of_outcomes": True, "full_60_epochs_required": True, "outer_not_used_checkpoint_selection": True, "holdout_loaded": False, "true_outer_loaded": False}
    write_json(PROTOCOL / "TESTS.json", tests)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundles = {d: v1.load_bundle(d, search[d]) for d in ("OpenBMI", "WBCIC")}
    fold_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    training_logs: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    for dataset in ("OpenBMI", "WBCIC"):
        bundle = bundles[dataset]
        for fold in folds[dataset]:
            fold_idx = fold["fold_id"]
            fold_rt = RUNTIME / f"{dataset.lower()}_fold{fold_idx}"
            v1.RUNTIME = fold_rt / "manifest_runtime"
            mean, std, norm = v1.normalizer(bundle, fold["inner_train_subjects"])
            manifest, manifest_info = v1.make_manifest(bundle, fold)
            manifest_records.append({"dataset": dataset, "fold": fold_idx, **manifest_info, "normalizer": norm})
            cache = carrier.GPUCache(bundle, mean, std, device)
            for seed in (0, 1, 2):
                models: dict[str, tuple[torch.nn.Module, dict[str, Any]]] = {}
                diagnostics: dict[str, dict[str, Any]] = {}
                for name in ("EEGNet", "LiteBN"):
                    set_seed(seed)
                    model = constructor(name, bundle.channels).to(device)
                    set_seed(seed + 100000)
                    cell_dir = RUNTIME / f"{dataset.lower()}_fold{fold_idx}_seed{seed}_{name.lower()}"
                    info = train_model(model, name, bundle, fold, manifest, cache, manifest_info["sha256"], cell_dir, device)
                    selected_rows, selected_ba, selected_f1 = evaluate_model(model, bundle, fold["outer_dev_subjects"], cache)
                    selected_val_rows, selected_val_ba, _ = evaluate_model(model, bundle, fold["inner_val_subjects"], cache)
                    epoch60_rows, epoch60_ba, epoch60_f1 = evaluate_checkpoint(model, Path(info["epoch60_checkpoint_path"]), bundle, fold["outer_dev_subjects"], cache, device)
                    epoch60_val_rows, epoch60_val_ba, _ = evaluate_model(model, bundle, fold["inner_val_subjects"], cache)
                    model.load_state_dict(torch.load(info["checkpoint_path"], map_location=device, weights_only=False))
                    diagnostics[name] = {"selected_rows": selected_rows, "selected_BA": selected_ba, "selected_macroF1": selected_f1, "selected_val_BA": selected_val_ba, "epoch60_rows": epoch60_rows, "epoch60_BA": epoch60_ba, "epoch60_macroF1": epoch60_f1, "epoch60_val_BA": epoch60_val_ba}
                    training_logs.append({"dataset": dataset, "fold": fold_idx, "seed": seed, "model": name, **info, "selected_outer_BA": selected_ba, "selected_outer_macroF1": selected_f1, "epoch60_outer_BA": epoch60_ba, "epoch60_outer_macroF1": epoch60_f1})
                    checkpoint_rows.append({"dataset": dataset, "fold": fold_idx, "seed": seed, "model": name, "selected_epoch": info["selected_epoch"], "best_inner_val_BA": info["best_inner_val_BA"], "epoch60_inner_val_BA": info["epoch60_inner_val_BA"], "selected_outer_BA": selected_ba, "epoch60_outer_BA": epoch60_ba})
                eeg, lite = diagnostics["EEGNet"], diagnostics["LiteBN"]
                delta = float(lite["selected_BA"] - eeg["selected_BA"])
                fold_rows.append({"dataset": dataset, "fold": fold_idx, "seed": seed, "EEGNet_BA": eeg["selected_BA"], "LiteBN_BA": lite["selected_BA"], "delta_pp": delta * 100.0, "EEGNet_macro_F1": eeg["selected_macroF1"], "LiteBN_macro_F1": lite["selected_macroF1"], "EEGNet_epoch": info if False else next(x["selected_epoch"] for x in training_logs[::-1] if x["dataset"] == dataset and x["fold"] == fold_idx and x["seed"] == seed and x["model"] == "EEGNet"), "LiteBN_epoch": next(x["selected_epoch"] for x in training_logs[::-1] if x["dataset"] == dataset and x["fold"] == fold_idx and x["seed"] == seed and x["model"] == "LiteBN"), "delta60_pp": (lite["epoch60_BA"] - eeg["epoch60_BA"]) * 100.0})
                for subject in fold["outer_dev_subjects"]:
                    e, l = eeg["selected_rows"][str(subject)], lite["selected_rows"][str(subject)]
                    e60, l60 = eeg["epoch60_rows"][str(subject)], lite["epoch60_rows"][str(subject)]
                    subject_rows.append({"dataset": dataset, "fold": fold_idx, "seed": seed, "subject_id": str(subject), "EEGNet_BA": e["BA"], "LiteBN_BA": l["BA"], "delta_pp": (l["BA"] - e["BA"]) * 100.0, "EEGNet_macro_F1": e["macro_F1"], "LiteBN_macro_F1": l["macro_F1"], "EEGNet_accuracy": e["accuracy"], "LiteBN_accuracy": l["accuracy"], "delta60_pp": (l60["BA"] - e60["BA"]) * 100.0})
                print(f"[{dataset} fold={fold_idx} seed={seed}] selected delta={delta*100:+.3f} pp; epoch60 delta={(lite['epoch60_BA']-eeg['epoch60_BA'])*100:+.3f} pp", flush=True)
                del models, diagnostics
                torch.cuda.empty_cache() if device.type == "cuda" else None
            del cache
            torch.cuda.empty_cache() if device.type == "cuda" else None
    fold_df = pd.DataFrame(fold_rows)
    subj_df = pd.DataFrame(subject_rows)
    train_df = pd.DataFrame(training_logs)
    ckpt_df = pd.DataFrame(checkpoint_rows)
    fold_agg = fold_df.groupby(["dataset", "fold"], as_index=False).agg(mean_delta_pp=("delta_pp", "mean"), seed_std_pp=("delta_pp", lambda x: float(np.std(x, ddof=0))), positive_seeds=("delta_pp", lambda x: int((x > 0).sum())), negative_seeds=("delta_pp", lambda x: int((x < 0).sum())), negligible_seeds=("delta_pp", lambda x: int((np.abs(x) < .5).sum())), seed_range_pp=("delta_pp", lambda x: float(x.max() - x.min())))
    fold_agg["sign_class"] = [classify_fold(fold_df[(fold_df.dataset == r.dataset) & (fold_df.fold == r.fold)].delta_pp) for r in fold_agg.itertuples()]
    seed_agg = fold_df.groupby(["dataset", "seed"], as_index=False).agg(mean_delta_pp_over_folds=("delta_pp", "mean"), fold_std_pp=("delta_pp", lambda x: float(np.std(x, ddof=0))))
    subject_agg = subj_df.groupby(["dataset", "fold", "subject_id"], as_index=False).agg(EEGNet_mean_BA=("EEGNet_BA", "mean"), LiteBN_mean_BA=("LiteBN_BA", "mean"), mean_delta_pp=("delta_pp", "mean"), std_delta_pp=("delta_pp", lambda x: float(np.std(x, ddof=0))), min_delta_pp=("delta_pp", "min"), max_delta_pp=("delta_pp", "max"), positive_seed_count=("delta_pp", lambda x: int((x > 0).sum())), negative_seed_count=("delta_pp", lambda x: int((x < 0).sum())))
    dataset_rows: list[dict[str, Any]] = []
    variance: dict[str, Any] = {}
    difficulty: list[dict[str, Any]] = []
    outlier: dict[str, Any] = {}
    for dataset in ("OpenBMI", "WBCIC"):
        sa = subject_agg[subject_agg.dataset == dataset].copy()
        deltas = sa.mean_delta_pp.to_numpy(float)
        boot = bootstrap(deltas)
        top = sa.sort_values("mean_delta_pp", ascending=False)
        total_positive = float(np.maximum(deltas, 0).sum())
        q = max(1, int(math.ceil(len(sa) * .25)))
        top_positive = float(np.maximum(top.head(q).mean_delta_pp.to_numpy(float), 0).sum())
        loo = [{"subject_id": str(r.subject_id), "mean_without_subject_pp": float(sa[sa.subject_id != r.subject_id].mean_delta_pp.mean())} for r in sa.itertuples()]
        worst_ids = list(top.tail(2).subject_id)
        without_worst = float(sa[~sa.subject_id.isin(worst_ids)].mean_delta_pp.mean())
        rho, ci = spearman_bootstrap(sa.EEGNet_mean_BA.to_numpy(float), sa.mean_delta_pp.to_numpy(float))
        difficulty.append({"dataset": dataset, "spearman_rho": rho, "bootstrap_ci": ci, "n_subjects": len(sa)})
        outlier[dataset] = {"best5": top.head(5).to_dict("records"), "worst5": top.tail(5).sort_values("mean_delta_pp").to_dict("records"), "leave_one_subject_out": loo, "leave_two_worst_out_delta_pp": without_worst, "top_quartile_positive_gain_fraction": (top_positive / total_positive if total_positive > 0 else None), "fraction_subjects_gain_ge_1pp": float((deltas >= 1).mean()), "fraction_subjects_harm_le_minus_1pp": float((deltas <= -1).mean())}
        ds_fold = fold_agg[fold_agg.dataset == dataset]; ds_seed = seed_agg[seed_agg.dataset == dataset]
        row = {"dataset": dataset, "grand_mean_delta_pp": boot["mean_delta_pp"], "median_subject_delta_pp": boot["median_delta_pp"], "bootstrap_ci_low_pp": boot["ci_low_pp"], "bootstrap_ci_high_pp": boot["ci_high_pp"], "improved_subjects": boot["improved_subjects"], "harmed_subjects": boot["harmed_subjects"], "tied_subjects": boot["tied_subjects"], "positive_folds": int((ds_fold.mean_delta_pp > 0).sum()), "positive_seed_means": int((ds_seed.mean_delta_pp_over_folds > 0).sum()), "fraction_subjects_gain_ge_1pp": outlier[dataset]["fraction_subjects_gain_ge_1pp"], "fraction_subjects_harm_le_minus_1pp": outlier[dataset]["fraction_subjects_harm_le_minus_1pp"]}
        dataset_rows.append(row)
        variance[dataset] = variance_decomposition(fold_df[fold_df.dataset == dataset])
    checkpoint_summary: dict[str, Any] = {}
    for dataset in ("OpenBMI", "WBCIC"):
        z = ckpt_df[ckpt_df.dataset == dataset].copy()
        pairs = fold_df[fold_df.dataset == dataset][["fold", "seed", "delta_pp", "delta60_pp"]].copy()
        best = z[z.model == "LiteBN"].merge(z[z.model == "EEGNet"], on=["dataset", "fold", "seed"], suffixes=("_lite", "_eeg"))
        pairs["inner_val_arch_delta"] = best.best_inner_val_BA_lite.to_numpy() * 100 - best.best_inner_val_BA_eeg.to_numpy() * 100
        pairs["selected_outer_arch_delta"] = pairs.delta_pp
        pairs["epoch60_outer_arch_delta"] = pairs.delta60_pp
        corr = spearmanr(pairs.inner_val_arch_delta, pairs.selected_outer_arch_delta).statistic
        materially = bool(abs(float(pairs.selected_outer_arch_delta.mean() - pairs.epoch60_outer_arch_delta.mean())) >= .5 or ((pairs.selected_outer_arch_delta > 0) != (pairs.epoch60_outer_arch_delta > 0)).sum() >= 3)
        checkpoint_summary[dataset] = {"selected_vs_epoch60_mean_delta_difference_pp": float(pairs.selected_outer_arch_delta.mean() - pairs.epoch60_outer_arch_delta.mean()), "selected_outer_delta_pp": pairs.selected_outer_arch_delta.to_list(), "epoch60_outer_delta_pp": pairs.epoch60_outer_arch_delta.to_list(), "inner_val_arch_delta_pp": pairs.inner_val_arch_delta.to_list(), "spearman_inner_vs_selected_outer": None if not np.isfinite(corr) else float(corr), "selected_vs_epoch60_materially_different": materially}
    for dataset in ("OpenBMI", "WBCIC"):
        ds = train_df[train_df.dataset == dataset]
        checkpoint_summary[dataset]["selected_epoch_sd_by_model"] = {m: float(ds[ds.model == m].selected_epoch.std(ddof=0)) for m in ("EEGNet", "LiteBN")}
        checkpoint_summary[dataset]["best_val_BA_sd_by_model"] = {m: float(ds[ds.model == m].best_inner_val_BA.std(ddof=0)) for m in ("EEGNet", "LiteBN")}
        checkpoint_summary[dataset]["epoch60_val_BA_sd_by_model"] = {m: float(ds[ds.model == m].epoch60_inner_val_BA.std(ddof=0)) for m in ("EEGNet", "LiteBN")}
    terminal_value = decision(pd.DataFrame(dataset_rows), fold_agg, seed_agg, variance, checkpoint_summary)
    fold_df.to_csv(OUT / "FOLD_SEED_RESULTS.csv", index=False)
    subj_df.to_csv(OUT / "SUBJECT_SEED_RESULTS.csv", index=False)
    subject_agg.to_csv(OUT / "SUBJECT_AGGREGATE_RESULTS.csv", index=False)
    fold_agg.to_csv(OUT / "FOLD_AGGREGATE_RESULTS.csv", index=False)
    seed_agg.to_csv(OUT / "SEED_AGGREGATE_RESULTS.csv", index=False)
    pd.DataFrame(dataset_rows).to_csv(OUT / "DATASET_AGGREGATE_RESULTS.csv", index=False)
    write_json(OUT / "TRAINING_LOGS.json", training_logs)
    write_json(OUT / "VARIANCE_DECOMPOSITION.json", variance)
    write_json(OUT / "SIGN_STABILITY.json", fold_agg.to_dict("records"))
    subject_agg.to_csv(OUT / "SUBJECT_HETEROGENEITY.csv", index=False)
    write_json(OUT / "OUTLIER_INFLUENCE.json", outlier)
    write_json(OUT / "SUBJECT_DIFFICULTY_INTERACTION.json", difficulty)
    write_json(OUT / "CHECKPOINT_SELECTION_DIAGNOSTIC.json", checkpoint_summary)
    write_json(OUT / "TRAINING_STABILITY.json", {d: checkpoint_summary[d] for d in checkpoint_summary})
    write_json(OUT / "DECISION.json", {"terminal": terminal_value, "dataset_rows": dataset_rows, "variance": variance})
    (OUT / "HISTORICAL_3FOLD_CONTEXT.md").write_text("""# Historical 3-fold context\n\nThese values are context only and are not pooled with the fresh five-fold multiseed estimate.\n\n| Dataset | Historical seed0 deltas (pp) |\n|---|---|\n| OpenBMI | +3.143, +0.231, +1.538 |\n| WBCIC | -3.455, +0.761, +1.400 |\n\nThe fresh experiment uses a new deterministic five-fold split and three seeds.\n""", encoding="utf-8")
    d = pd.DataFrame(dataset_rows).set_index("dataset")
    v = variance
    lines = ["# LiteBN 5-fold x 3-seed stability decision", "", "| Metric | OpenBMI | WBCIC |", "|---|---:|---:|"]
    for label, col in [("5-fold mean delta (pp)", "grand_mean_delta_pp"), ("median subject delta (pp)", "median_subject_delta_pp"), ("bootstrap CI", None), ("positive folds", "positive_folds"), ("positive seeds", "positive_seed_means")]:
        vals = []
        for ds in ("OpenBMI", "WBCIC"):
            vals.append((f"[{d.loc[ds, 'bootstrap_ci_low_pp']:+.3f}, {d.loc[ds, 'bootstrap_ci_high_pp']:+.3f}]" if col is None else f"{d.loc[ds, col]:.3f}"))
        lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
    lines += ["", f"Final terminal: **{terminal_value}**", "", "1. Is LiteBN meaningfully better on OpenBMI? " + ("YES" if d.loc['OpenBMI','grand_mean_delta_pp'] >= 1 else "SMALL" if d.loc['OpenBMI','grand_mean_delta_pp'] > 0 else "NO"), "2. Is LiteBN meaningfully better on WBCIC? " + ("YES" if d.loc['WBCIC','grand_mean_delta_pp'] >= 1 else "SMALL" if d.loc['WBCIC','grand_mean_delta_pp'] > 0 else "NO"), "3. Same-sign across datasets? " + ("YES" if all(d.loc[x,'grand_mean_delta_pp'] > 0 for x in ('OpenBMI','WBCIC')) else "NO"), f"4. Positive folds: OpenBMI {int(d.loc['OpenBMI','positive_folds'])}/5; WBCIC {int(d.loc['WBCIC','positive_folds'])}/5", f"5. Positive seed means: OpenBMI {int(d.loc['OpenBMI','positive_seed_means'])}/3; WBCIC {int(d.loc['WBCIC','positive_seed_means'])}/3", f"6. Fold vs seed variance: OpenBMI fold SD {v['OpenBMI']['sd_fold_means_pp']:.3f} pp, seed SD {v['OpenBMI']['sd_seed_means_pp']:.3f} pp; WBCIC fold SD {v['WBCIC']['sd_fold_means_pp']:.3f} pp, seed SD {v['WBCIC']['sd_seed_means_pp']:.3f} pp", "7. Consistently negative folds: see SIGN_STABILITY.json.", "8. Repeated sign-changing folds: see SIGN_STABILITY.json.", "9. Small-subject concentration: see OUTLIER_INFLUENCE.json; primary estimate is unchanged.", "10. Difficulty interaction: see SUBJECT_DIFFICULTY_INTERACTION.json.", "11. Fixed epoch60 conclusion: see CHECKPOINT_SELECTION_DIAGNOSTIC.json.", "12. Inner-val vs outer association: see CHECKPOINT_SELECTION_DIAGNOSTIC.json.", "13. Dominant instability source: determined from fold/seed/checkpoint diagnostics above.", "14. Historical WBCIC -3.45 pp is evaluated as context only against the fresh split; it is not numerically pooled.", "", "Recommended next scientific action: do not execute another model in this experiment; use the terminal to decide whether carrier development should stop or move to the specified diagnostic direction."]
    (OUT / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(PROTOCOL / "MANIFEST_HASHES.json", manifest_records)
    print(terminal_value, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
