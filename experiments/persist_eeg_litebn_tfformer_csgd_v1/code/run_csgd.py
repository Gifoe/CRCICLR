#!/usr/bin/env python3
"""Frozen LiteBN/TFFormer cross-session generalization-drop analysis.

This script is inference-only. It reuses the exact selected checkpoints and
fold-specific train-only normalizers from the final multiseed evaluation.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


REPO = Path(os.environ.get("CSGD_REPO", "/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK")).resolve()
EXP = REPO / "experiments/persist_eeg_litebn_tfformer_csgd_v1"
OUT = EXP / "outputs/csgd_v1"
PROTOCOL = EXP / "protocol"
MULTI_EXP = REPO / "experiments/persist_eeg_litebn_tfformer_multiseed_v1"
NORMALIZERS = Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime/normalizers")
CARRIER = Path("/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")
TASK_RUNTIME = Path("/root/rivermind-data/openbmi_task_generality_runtime")
WBCIC_OUTER = Path("/root/rivermind-data/persist_eeg_cache/wbcic_true_outer_v1/wbcic_epochs")

TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
MODELS = ("LiteBN", "TFFormer")
SEEDS = (0, 1, 2)
FOLDS = tuple(range(5))
OPENBMI_SUBJECTS = ("4", "12", "13", "17", "18", "24", "25", "29", "36", "37", "39", "42", "51", "54")
WBCIC_SUBJECTS = ("sub-4", "sub-8", "sub-10", "sub-15", "sub-20", "sub-39", "sub-40", "sub-43", "sub-46", "sub-51")
BOOTSTRAP_DRAWS = 20_000
BATCH = int(os.environ.get("CSGD_BATCH", "64"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    pd.DataFrame(list(rows)).to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_runtime():
    # Importing this module defines the exact final model/runtime but does not run training.
    os.environ["TFFREM_SEED"] = "1"
    code = MULTI_EXP / "code"
    sys.path.insert(0, str(code))
    source = code / "run_multiseed.py"
    spec = importlib.util.spec_from_file_location("csgd_frozen_runtime", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen TFFormer runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def litebn_path(task: str, seed: int, fold: int) -> Path:
    if task in ("OpenBMI_ERP", "OpenBMI_SSVEP"):
        short = "erp" if task.endswith("ERP") else "ssvep"
        return TASK_RUNTIME / f"{short}_fold{fold}_seed{seed}_litebn/selected_best.pt"
    short = "openbmi" if task == "OpenBMI_MI" else "wbcic"
    return CARRIER / f"{short}_fold{fold}_seed{seed}_litebn/selected_best.pt"


def tfformer_path(task: str, seed: int, fold: int) -> Path:
    if seed == 0:
        if task == "OpenBMI_SSVEP":
            return REPO / f"experiments/persist_eeg_litebn_tfformer_v1/runtime/checkpoints/OpenBMI_SSVEP/fold{fold}.pt"
        return REPO / f"experiments/persist_eeg_litebn_tfformer_remaining_tasks_seed0_v1/runtime/checkpoints/{task}/fold{fold}/selected.pt"
    return MULTI_EXP / f"runtime/seed{seed}/checkpoints/{task}/fold{fold}/selected.pt"


def normalizer_path(task: str, fold: int) -> Path:
    return NORMALIZERS / f"{task.lower()}_fold{fold}.npz"


def build_wbcic_outer_bundle(runtime):
    rows = []
    inventory = []
    for subject in WBCIC_SUBJECTS:
        for session in (0, 1, 2):
            signal = WBCIC_OUTER / subject / f"ses-{session}_epochs.npy"
            labels = WBCIC_OUTER / subject / f"ses-{session}_labels.npy"
            if not signal.is_file() or not labels.is_file():
                raise FileNotFoundError(f"missing WBCIC true-outer session: {signal} / {labels}")
            x = np.load(signal, mmap_mode="r", allow_pickle=False)
            y = np.load(labels, mmap_mode="r", allow_pickle=False)
            if x.shape != (200, 58, 1000) or x.dtype != np.float16:
                raise RuntimeError(f"WBCIC signal schema mismatch: {signal}: {x.shape}/{x.dtype}")
            if y.shape != (200,) or set(map(int, np.unique(y))) != {0, 1}:
                raise RuntimeError(f"WBCIC label schema mismatch: {labels}: {y.shape}/{y.dtype}")
            inventory.append({
                "task": "WBCIC_MI", "subject_id": subject, "session": f"S{session}",
                "trials": int(len(y)), "signal_path": str(signal), "label_path": str(labels),
                "signal_shape": "200x58x1000", "signal_dtype": str(x.dtype), "label_dtype": str(y.dtype),
            })
            rows.extend(runtime.base.Row(subject, session, str(signal), index, int(label)) for index, label in enumerate(y))
    return runtime.base.SignalBundle("WBCIC_MI", WBCIC_SUBJECTS, rows), inventory


def subjects_and_sessions(task: str):
    if task == "WBCIC_MI":
        return WBCIC_SUBJECTS, (0, 1, 2)
    return OPENBMI_SUBJECTS, (1, 2)


def evaluate_sessions(runtime, model, model_name: str, bundle, raw, spectral, subjects, sessions, mean, std):
    model.eval()
    output = []
    with torch.inference_mode():
        for subject in subjects:
            for session in sessions:
                indices = bundle.indices([subject], [session])
                if len(indices) == 0:
                    raise RuntimeError(f"empty evaluation cell: {bundle.task}/{subject}/S{session}")
                logits = []
                for start in range(0, len(indices), BATCH):
                    current = indices[start:start + BATCH]
                    if model_name == "LiteBN":
                        value, _ = raw.batch(current, mean, std)
                        logits.append(model(value)[0].float().cpu().numpy())
                    else:
                        value, _, cached = spectral.batch(current)
                        logits.append(model(value, cached)[0].float().cpu().numpy())
                metrics = runtime.base.classification_metrics(bundle.labels(indices), np.concatenate(logits, axis=0))
                output.append({
                    "subject_id": str(subject), "session": f"S{session}", "trials": int(len(indices)),
                    "BA": float(metrics["BA"]), "macro_F1": float(metrics["macro_F1"]),
                })
    return output


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    # Subject is the resampling unit. Chunking bounds transient memory.
    means = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 2_000):
        stop = min(start + 2_000, BOOTSTRAP_DRAWS)
        draws = generator.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def aggregate(session_rows: list[dict[str, Any]]):
    frame = pd.DataFrame(session_rows)
    expected = len(FOLDS) * len(SEEDS)
    grouped = frame.groupby(["task", "model", "subject_id", "session"], as_index=False).agg(
        BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), valid_fold_seed_evaluations=("BA", "size")
    )
    if not (grouped["valid_fold_seed_evaluations"] == expected).all():
        bad = grouped[grouped["valid_fold_seed_evaluations"] != expected]
        raise RuntimeError(f"incomplete fold-seed cells:\n{bad.to_string(index=False)}")

    subject_rows = []
    for (task, model, subject), part in grouped.groupby(["task", "model", "subject_id"], sort=False):
        cells = {str(row.session): row for row in part.itertuples(index=False)}
        required = ("S0", "S1", "S2") if task == "WBCIC_MI" else ("S1", "S2")
        if set(cells) != set(required):
            raise RuntimeError(f"session coverage mismatch: {task}/{model}/{subject}: {sorted(cells)}")
        if task == "WBCIC_MI":
            source_ba = (cells["S0"].BA + cells["S1"].BA) / 2.0
            source_f1 = (cells["S0"].macro_F1 + cells["S1"].macro_F1) / 2.0
        else:
            source_ba, source_f1 = cells["S1"].BA, cells["S1"].macro_F1
        future_ba, future_f1 = cells["S2"].BA, cells["S2"].macro_F1
        row = {
            "task": task, "model": model, "subject_id": subject,
            "valid_fold_seed_evaluations_per_session": expected,
            "source_BA": float(source_ba), "future_BA": float(future_ba),
            "source_macro_F1": float(source_f1), "future_macro_F1": float(future_f1),
            "CSGD_pp": float(100.0 * (source_ba - future_ba)),
        }
        for session in required:
            row[f"{session}_BA"] = float(cells[session].BA)
            row[f"{session}_macro_F1"] = float(cells[session].macro_F1)
        if task == "WBCIC_MI":
            row["S0_to_S2_drop_pp"] = float(100.0 * (cells["S0"].BA - cells["S2"].BA))
            row["S1_to_S2_drop_pp"] = float(100.0 * (cells["S1"].BA - cells["S2"].BA))
        subject_rows.append(row)

    task_rows = []
    subject_frame = pd.DataFrame(subject_rows)
    for (task, model), part in subject_frame.groupby(["task", "model"], sort=False):
        values = part["CSGD_pp"].to_numpy(np.float64)
        key = int.from_bytes(hashlib.sha256(f"{task}/{model}/CSGD".encode()).digest()[:4], "little")
        low, high = bootstrap_mean(values, key)
        row = {
            "task": task, "model": model, "biological_subjects": int(len(part)),
            "source_BA": float(part["source_BA"].mean()), "future_BA": float(part["future_BA"].mean()),
            "source_macro_F1": float(part["source_macro_F1"].mean()),
            "future_macro_F1": float(part["future_macro_F1"].mean()),
            "CSGD_mean_pp": float(values.mean()), "CSGD_median_pp": float(np.median(values)),
            "CSGD_CI95_low_pp": low, "CSGD_CI95_high_pp": high,
            "fraction_CSGD_gt_0": float(np.mean(values > 0.0)),
            "fraction_CSGD_gt_1pp": float(np.mean(values > 1.0)),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        }
        if task == "WBCIC_MI":
            for column in ("S0_BA", "S1_BA", "S2_BA", "S0_macro_F1", "S1_macro_F1", "S2_macro_F1", "S0_to_S2_drop_pp", "S1_to_S2_drop_pp"):
                row[column] = float(part[column].mean())
        task_rows.append(row)

    paired_rows = []
    for task in TASKS:
        lite = subject_frame[(subject_frame.task == task) & (subject_frame.model == "LiteBN")][["subject_id", "CSGD_pp"]].rename(columns={"CSGD_pp": "LiteBN_CSGD_pp"})
        tf = subject_frame[(subject_frame.task == task) & (subject_frame.model == "TFFormer")][["subject_id", "CSGD_pp"]].rename(columns={"CSGD_pp": "TFFormer_CSGD_pp"})
        pair = lite.merge(tf, on="subject_id", how="inner", validate="one_to_one")
        expected_subjects = len(WBCIC_SUBJECTS if task == "WBCIC_MI" else OPENBMI_SUBJECTS)
        if len(pair) != expected_subjects:
            raise RuntimeError(f"paired subject mismatch for {task}: {len(pair)} != {expected_subjects}")
        delta = (pair["TFFormer_CSGD_pp"] - pair["LiteBN_CSGD_pp"]).to_numpy(np.float64)
        key = int.from_bytes(hashlib.sha256(f"{task}/paired/CSGD".encode()).digest()[:4], "little")
        low, high = bootstrap_mean(delta, key)
        paired_rows.append({
            "task": task, "biological_subjects": int(len(pair)),
            "LiteBN_CSGD_mean_pp": float(pair["LiteBN_CSGD_pp"].mean()),
            "TFFormer_CSGD_mean_pp": float(pair["TFFormer_CSGD_pp"].mean()),
            "TFFormer_minus_LiteBN_CSGD_mean_pp": float(delta.mean()),
            "delta_CI95_low_pp": low, "delta_CI95_high_pp": high,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        })
    return subject_rows, task_rows, paired_rows


def make_report(task_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Frozen LiteBN / TFFormer cross-session generalization drop", "",
        "No model was trained, fine-tuned, recalibrated, adapted, or used with updated BN statistics. "
        "Each selected checkpoint used its original fold-specific train-only channelwise normalizer unchanged for every session.", "",
        "OpenBMI uses the previously exposed 14-subject internal-heldout diagnostic cohort on S1 and S2. "
        "WBCIC uses the previously exposed 10-subject true-outer cohort on S0, S1, and S2. "
        "Labels were used only for post-hoc metrics; these cohorts are not untouched after this analysis.", "",
        "## Task-level CSGD", "",
        "| Task | Model | Source BA | Future BA | CSGD pp [95% CI] | Median CSGD | Subjects CSGD >1 pp |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in task_rows:
        lines.append(
            f"| {row['task']} | {row['model']} | {100*row['source_BA']:.2f}% | {100*row['future_BA']:.2f}% | "
            f"{row['CSGD_mean_pp']:.2f} [{row['CSGD_CI95_low_pp']:.2f}, {row['CSGD_CI95_high_pp']:.2f}] | "
            f"{row['CSGD_median_pp']:.2f} | {100*row['fraction_CSGD_gt_1pp']:.1f}% |"
        )
    lines += ["", "## Paired LiteBN vs TFFormer", "", "| Task | LiteBN CSGD | TFFormer CSGD | TFFormer - LiteBN CSGD [95% CI] |", "|---|---:|---:|---:|"]
    for row in paired_rows:
        lines.append(
            f"| {row['task']} | {row['LiteBN_CSGD_mean_pp']:.2f} | {row['TFFormer_CSGD_mean_pp']:.2f} | "
            f"{row['TFFormer_minus_LiteBN_CSGD_mean_pp']:.2f} [{row['delta_CI95_low_pp']:.2f}, {row['delta_CI95_high_pp']:.2f}] |"
        )
    lines += [
        "", "## WBCIC session details", "",
        "| Model | S0 BA | S1 BA | S2 BA | S0->S2 drop | S1->S2 drop |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in task_rows:
        if row["task"] == "WBCIC_MI":
            lines.append(f"| {row['model']} | {100*row['S0_BA']:.2f}% | {100*row['S1_BA']:.2f}% | {100*row['S2_BA']:.2f}% | {row['S0_to_S2_drop_pp']:.2f} | {row['S1_to_S2_drop_pp']:.2f} |")
    lines += [
        "", "CSGD is computed after first averaging all 15 fold-seed evaluations within each biological subject. "
        "Confidence intervals use 20,000 bootstrap resamples of biological subjects, never folds or seeds.", "",
        "Smaller CSGD means less future-session degradation. Negative paired delta means TFFormer degrades less. "
        "CSGD alone does not establish model superiority; it must be interpreted jointly with future-session BA and Macro-F1.",
    ]
    return "\n".join(lines)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tractable frozen TFFormer inference")
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("CSGD_GPU_FRACTION", "0.45")), 0)
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime()
    script_hash = sha256(Path(__file__))

    checkpoint_audit = []
    for task in TASKS:
        for fold in FOLDS:
            for seed in SEEDS:
                for model, path in (("LiteBN", litebn_path(task, seed, fold)), ("TFFormer", tfformer_path(task, seed, fold))):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    checkpoint_audit.append({"task": task, "model": model, "fold": fold, "seed": seed, "path": str(path), "sha256": sha256(path)})
    normalizer_audit = []
    for task in TASKS:
        for fold in FOLDS:
            path = normalizer_path(task, fold)
            if not path.is_file():
                raise FileNotFoundError(path)
            mean, std, metadata = runtime.base.load_tensor_pair(path)
            if len(mean) != int(runtime.base.TASKS[task]["channels"]) or len(std) != len(mean):
                raise RuntimeError(f"normalizer shape mismatch: {path}")
            normalizer_audit.append({
                "task": task, "fold": fold, "path": str(path), "file_sha256": sha256(path),
                "mean_std_sha256": metadata.get("mean_std_sha256"), "training_subjects": json.dumps(metadata.get("subjects")),
                "source_sessions": json.dumps(metadata.get("source_sessions")), "train_only_frozen": True,
            })
    atomic_csv(PROTOCOL / "CHECKPOINT_AUDIT.csv", checkpoint_audit)
    atomic_csv(PROTOCOL / "NORMALIZER_AUDIT.csv", normalizer_audit)
    atomic_json(PROTOCOL / "CSGD_PROTOCOL_LOCK.json", {
        "analysis": "frozen post-hoc cross-session generalization drop",
        "script_sha256": script_hash, "tasks": TASKS, "models": MODELS, "folds": FOLDS, "seeds": SEEDS,
        "openbmi_subjects": OPENBMI_SUBJECTS, "openbmi_sessions": [1, 2],
        "wbcic_subjects": WBCIC_SUBJECTS, "wbcic_sessions": [0, 1, 2],
        "checkpoint_selection_changed": False, "training": False, "finetuning": False,
        "recalibration": False, "target_adaptation": False, "BN_updates": False,
        "normalization": "exact original fold-specific train-only channelwise normalizer, unchanged across evaluation sessions",
        "decoder": "frozen model classifier head; no ridge probe",
        "aggregation": "average folds and seeds within biological subject, then bootstrap subjects",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "cohort_status": "previously accessed diagnostic/evaluation cohorts; labels used only for post-hoc metrics",
    })

    result_path = OUT / "SESSION_SUBJECT_RESULTS.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.is_file() else []
    complete = {(str(r["task"]), str(r["model"]), int(r["fold"]), int(r["seed"])) for r in rows}
    data_inventory = []

    for task in TASKS:
        subjects, sessions = subjects_and_sessions(task)
        if task == "WBCIC_MI":
            bundle, inventory = build_wbcic_outer_bundle(runtime)
            data_inventory.extend(inventory)
        else:
            bundle = runtime.base.build_bundle(task, subjects)
            for subject in subjects:
                for session in sessions:
                    ids = bundle.indices([subject], [session])
                    data_inventory.append({"task": task, "subject_id": subject, "session": f"S{session}", "trials": int(len(ids)), "signal_path": "canonical OpenBMI cache", "label_path": "canonical OpenBMI cache", "signal_shape": f"{len(ids)}x{bundle.channels}x{bundle.samples}", "signal_dtype": "float32", "label_dtype": "canonical mapped integer"})
        print(f"CSGD_DATA task={task} rows={len(bundle.rows)} device={device}", flush=True)
        raw = runtime.base.RawGPUCache(bundle, device)
        for fold in FOLDS:
            mean, std, normalizer_metadata = runtime.base.load_tensor_pair(normalizer_path(task, fold))
            # LiteBN does not need the expensive spectral cache.
            for seed in SEEDS:
                key = (task, "LiteBN", fold, seed)
                if key in complete:
                    continue
                path = litebn_path(task, seed, fold)
                model = runtime.base.build_model("LiteBN_BASELINE", task)
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=True)
                model = model.to(device).eval()
                current = evaluate_sessions(runtime, model, "LiteBN", bundle, raw, None, subjects, sessions, mean, std)
                for row in current:
                    row.update({"task": task, "model": "LiteBN", "fold": fold, "seed": seed, "checkpoint_path": str(path), "checkpoint_sha256": sha256(path), "normalizer_path": str(normalizer_path(task, fold)), "normalizer_mean_std_sha256": normalizer_metadata.get("mean_std_sha256")})
                rows.extend(current)
                atomic_csv(result_path, rows)
                complete.add(key)
                print(f"CSGD_DONE task={task} model=LiteBN fold={fold} seed={seed}", flush=True)
                del model
                gc.collect(); torch.cuda.empty_cache()

            pending_tf = [seed for seed in SEEDS if (task, "TFFormer", fold, seed) not in complete]
            if pending_tf:
                spectral = runtime.Cache(raw, mean, std, device)
                print(f"CSGD_SPECTRAL task={task} fold={fold} GiB={spectral.bytes/2**30:.3f}", flush=True)
                for seed in pending_tf:
                    path = tfformer_path(task, seed, fold)
                    baseline = runtime.base.build_model("LiteBN_BASELINE", task)
                    model = runtime.CachedTFFormer(baseline, 250)
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    model.load_state_dict(payload["state_dict"], strict=True)
                    model = model.to(device).eval()
                    current = evaluate_sessions(runtime, model, "TFFormer", bundle, raw, spectral, subjects, sessions, mean, std)
                    for row in current:
                        row.update({"task": task, "model": "TFFormer", "fold": fold, "seed": seed, "checkpoint_path": str(path), "checkpoint_sha256": sha256(path), "normalizer_path": str(normalizer_path(task, fold)), "normalizer_mean_std_sha256": normalizer_metadata.get("mean_std_sha256")})
                    rows.extend(current)
                    atomic_csv(result_path, rows)
                    complete.add((task, "TFFormer", fold, seed))
                    print(f"CSGD_DONE task={task} model=TFFormer fold={fold} seed={seed}", flush=True)
                    del model, baseline
                    gc.collect(); torch.cuda.empty_cache()
                del spectral
                gc.collect(); torch.cuda.empty_cache()
        del raw, bundle
        gc.collect(); torch.cuda.empty_cache()
        print(f"CSGD_TASK_DONE task={task}", flush=True)

    atomic_csv(PROTOCOL / "EVALUATION_DATA_INVENTORY.csv", data_inventory)
    subject_rows, task_rows, paired_rows = aggregate(rows)
    atomic_csv(OUT / "SUBJECT_LEVEL_CSGD.csv", subject_rows)
    atomic_csv(OUT / "TASK_LEVEL_CSGD.csv", task_rows)
    atomic_csv(OUT / "LITEBN_TFFORMER_PAIRED_CSGD.csv", paired_rows)
    atomic_text(OUT / "FINAL_CSGD_REPORT.md", make_report(task_rows, paired_rows))
    atomic_json(OUT / "CSGD_COMPLETION.json", {
        "status": "COMPLETE", "session_subject_rows": len(rows), "subject_rows": len(subject_rows),
        "task_rows": len(task_rows), "paired_rows": len(paired_rows), "training_or_adaptation": False,
        "checkpoint_audit_rows": len(checkpoint_audit), "normalizer_audit_rows": len(normalizer_audit),
    })
    print("CSGD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
