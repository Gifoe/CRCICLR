"""Supporting PERSIST-CF rescue/harm decomposition with a filtered loader.

This is not an AGDI authorization gate.  It reproduces the frozen DDA-A model
and exact matched-random construction, but adds correctness-transition and
signed task-consequence summaries.  Only frozen OpenBMI outer-TRAIN subjects
for each fold are returned by the parquet loader and materialized from h0.
"""

from __future__ import annotations

import argparse
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


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
EXP_ROOT = REPO_ROOT / "experiments" / "persist_eeg_external_actionability_v1"
OUT = EXP_ROOT / "outputs"
RESULTS = OUT / "results"
REFERENCE_DDA = REPO_ROOT / "experiments" / "persist_eeg_dda_v1"
P5_ROOT = REPO_ROOT / "experiments" / "persist_eeg_p5_icg"
CF_ROOT = REPO_ROOT / "experiments" / "persist_eeg_cf"
ROUTER_ROOT = REPO_ROOT / "experiments" / "persist_eeg_router"
STAGE0_META = REPO_ROOT / "outputs" / "persist_eeg_stage0" / "embeddings" / "eegnet" / "fold-0" / "seed-0" / "metadata.parquet"
IMPLEMENTATION_ID = "persist_cf_rescue_harm_supporting_20260817"
RANDOM_DRAWS = 100
BOOTSTRAP_DRAWS = 5_000
EPS = 1e-12


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DDA = import_file("external_supporting_frozen_dda", REFERENCE_DDA / "code" / "persist_dda_v1.py")
P5 = DDA.P5
CF = DDA.CF
ROUTER = DDA.ROUTER


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32 - 1)


def balanced_accuracy(y: np.ndarray, prediction: np.ndarray) -> float:
    truth, pred = np.asarray(y, dtype=np.int64), np.asarray(prediction, dtype=np.int64)
    values = [float(np.mean(pred[truth == label] == label)) for label in np.unique(truth)]
    return float(np.mean(values)) if values else float("nan")


def structural_mi_index() -> np.ndarray:
    """Return anonymous full-manifest positions of MI rows, no IDs/labels."""
    frame = pd.read_parquet(STAGE0_META, columns=["paradigm", "embedding_index"])
    positions = frame.loc[frame.paradigm.astype(str) == "mi", "embedding_index"].to_numpy(dtype=np.int64)
    del frame
    if len(positions) != 10_800 or len(np.unique(positions)) != len(positions):
        raise RuntimeError("Invalid anonymous MI structural index")
    return positions


def filtered_meta(subjects: Sequence[str], mi_positions: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    allowed = sorted(set(map(str, subjects)), key=int)
    source = pd.read_parquet(
        P5.MANIFEST,
        filters=[("paradigm", "==", "mi"), ("subject_id", "in", allowed)],
    )
    alignment = pd.read_parquet(
        STAGE0_META,
        columns=["subject_id", "paradigm", "embedding_index"],
        filters=[("paradigm", "==", "mi"), ("subject_id", "in", allowed)],
    )
    if len(source) != len(alignment) or source.subject_id.astype(str).tolist() != alignment.subject_id.astype(str).tolist():
        raise RuntimeError("DATA_SCOPE_VIOLATION: filtered manifest/alignment mismatch")
    if set(source.subject_id.astype(str)) != set(allowed):
        raise RuntimeError("DATA_SCOPE_VIOLATION: filtered manifest returned an invalid subject set")
    ordinal = np.searchsorted(mi_positions, alignment.embedding_index.to_numpy(dtype=np.int64))
    if np.any(ordinal < 0) or np.any(ordinal >= len(mi_positions)) or not np.array_equal(mi_positions[ordinal], alignment.embedding_index.to_numpy(dtype=np.int64)):
        raise RuntimeError("Filtered h0 ordinal mapping failed")
    meta = source.copy().reset_index(drop=True)
    labels = sorted(meta.event_label.astype(str).unique().tolist())
    if labels != ["left_hand", "right_hand"]:
        raise RuntimeError(f"Unexpected filtered MI labels: {labels}")
    mapper = {labels[0]: 0, labels[1]: 1}
    meta["manifest_index"] = alignment.embedding_index.to_numpy(dtype=np.int64)
    meta["subject"] = meta.subject_id.astype(str)
    meta["session"] = meta.session_id.astype(str)
    meta["label"] = meta.event_label.astype(str).map(mapper).astype(np.int64)
    return meta, ordinal


def filtered_run(fold: int, seed: int, subjects: Sequence[str], mi_positions: np.ndarray):
    meta, ordinal = filtered_meta(subjects, mi_positions)
    h_path = P5.OUT / "cache" / f"fold-{fold}" / f"seed-{seed}" / "h0.npy"
    h_container = np.load(h_path, mmap_mode="r", allow_pickle=False)
    if h_container.shape != (10_800, 128):
        raise RuntimeError(f"Invalid h0 container {h_path}: {h_container.shape}")
    h = np.asarray(h_container[ordinal], dtype=np.float32)
    del h_container
    if h.shape != (len(meta), 128) or not np.isfinite(h).all():
        raise RuntimeError("Invalid filtered h0 materialization")
    art = P5.load_artifacts(fold, seed)
    q = P5.q_from_h(h, art)
    split = {"train_subjects": sorted(set(map(str, subjects)), key=int)}
    return ROUTER.AuditRun(fold=fold, seed=seed, meta=meta, h=h, q=q, art=art, split=split)


def positions(meta: pd.DataFrame, subjects: Sequence[str]) -> np.ndarray:
    wanted = set(map(str, subjects))
    return np.flatnonzero(meta.subject.astype(str).isin(wanted).to_numpy())


def fit_model(run: Any, cfg: Any, train_pos: np.ndarray, tag: str, device: torch.device) -> Any:
    cache = OUT / "cache" / "cf_geometry_targets" / f"fold-{run.fold}" / f"seed-{run.seed}" / f"{tag}.npz"
    targets = P5.build_geometry_targets(run.meta, run.q, train_pos, run.art, cache)
    # Match DDA.fit_v2_control's frozen initialization and sampler streams
    # exactly; only the loader/cache location differs.
    model = ROUTER.initialise_v2_control(run, cfg, targets, f"dda-{tag}", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-3)
    train_meta = run.meta.iloc[train_pos].reset_index(drop=True)
    h = torch.as_tensor(np.asarray(run.h[train_pos], dtype=np.float32), device=device)
    q = torch.as_tensor(np.asarray(run.q[train_pos], dtype=np.float32), device=device)
    y = torch.as_tensor(train_meta.label.to_numpy(dtype=np.int64), dtype=torch.long, device=device)
    sampler = P5.StructuredSampler(train_meta, train_meta.subject.unique().tolist(), subjects_per_batch=6, trials_per_class=4)
    for epoch in range(int(cfg.epochs)):
        model.train()
        batches = sampler.batches(epoch, DDA.stable_seed(DDA.IMPLEMENTATION_ID, run.fold, run.seed, tag, "sampler"))
        for batch in batches:
            index = torch.as_tensor(batch, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, delta = model(h.index_select(0, index), q.index_select(0, index))
            loss = F.cross_entropy(logits, y.index_select(0, index)) + cfg.lambda_drift * P5.drift_loss(delta, run.art)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model.eval()


def metric_rows(
    clean_logits: np.ndarray, shifted_logits: np.ndarray, labels: np.ndarray,
    subjects: np.ndarray, info: Mapping[str, Any],
) -> list[dict[str, Any]]:
    base = np.asarray(clean_logits, dtype=np.float64)
    shifted = np.asarray(shifted_logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    base_probability = DDA.softmax(base)
    shifted_probability = DDA.softmax(shifted)
    base_prediction = base.argmax(1)
    shifted_prediction = shifted.argmax(2)
    base_correct = base_prediction == y
    shifted_correct = shifted_prediction == y[None, :]
    base_ce = DDA.ce_per_row(base, y)
    shifted_ce = DDA.ce_per_row(shifted, y[None, :])
    base_margin = DDA.true_margin(base, y)
    shifted_margin = DDA.true_margin(shifted, y[None, :])
    base_true_probability = base_probability[np.arange(len(y)), y]
    shifted_true_probability = np.take_along_axis(shifted_probability, y[None, :, None], axis=2)[..., 0]
    rows = []
    subject_values = np.asarray(subjects, dtype=str)
    for subject in sorted(set(subject_values), key=int):
        index = np.flatnonzero(subject_values == subject)
        rescue = (~base_correct[index])[None, :] & shifted_correct[:, index]
        harm = base_correct[index][None, :] & (~shifted_correct[:, index])
        stable_correct = base_correct[index][None, :] & shifted_correct[:, index]
        stable_wrong = (~base_correct[index])[None, :] & (~shifted_correct[:, index])
        ba_delta = [
            balanced_accuracy(y[index], shifted_prediction[shift, index]) - balanced_accuracy(y[index], base_prediction[index])
            for shift in range(len(shifted))
        ]
        confidence_delta = shifted_probability[:, index].max(axis=2) - base_probability[index].max(axis=1)[None, :]
        rows.append({
            **dict(info), "subject": subject, "n_trials": int(len(index)), "n_shifts": int(len(shifted)),
            "rescue_rate": float(rescue.mean()), "harm_rate": float(harm.mean()),
            "net_rescue": float(rescue.mean() - harm.mean()),
            "stable_correct_rate": float(stable_correct.mean()), "stable_wrong_rate": float(stable_wrong.mean()),
            "true_margin_change": float(np.mean(shifted_margin[:, index] - base_margin[index][None, :])),
            "correct_probability_change": float(np.mean(shifted_true_probability[:, index] - base_true_probability[index][None, :])),
            "prediction_flip_rate": float(np.mean(shifted_prediction[:, index] != base_prediction[index][None, :])),
            "ce_change": float(np.mean(shifted_ce[:, index] - base_ce[index][None, :])),
            "ba_change": float(np.mean(ba_delta)), "confidence_change": float(np.mean(confidence_delta)),
        })
    return rows


def hierarchical_bootstrap(frame: pd.DataFrame, column: str, seed: int) -> np.ndarray:
    runs = sorted(frame.run.unique().tolist())
    grouped = {run: frame[frame.run == run][column].to_numpy(dtype=np.float64) for run in runs}
    rng = np.random.default_rng(int(seed))
    output = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        selected_runs = rng.choice(runs, size=len(runs), replace=True)
        run_values = []
        for run in selected_runs:
            values = grouped[str(run)]
            run_values.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        output[draw] = float(np.mean(run_values))
    return output


def run(device: torch.device) -> dict[str, Any]:
    dda_result = json.loads((REFERENCE_DDA / "outputs" / "results" / "DDA_A_RESULT.json").read_text(encoding="utf-8"))
    if dda_result.get("status") != "DDA_A_FAIL":
        raise RuntimeError("Frozen DDA-A status is not the required FAIL")
    stress = json.loads((CF_ROOT / "outputs" / "protocol" / "STRESS_BANK_FREEZE.json").read_text(encoding="utf-8"))
    bases = ROUTER.selected_bases()
    mi_positions = structural_mi_index()
    run_cache: dict[tuple[int, int], Any] = {}
    real_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    materialized: dict[str, Any] = {}
    for bank_index, bank in enumerate(stress["banks"]):
        fold, seed, inner = int(bank["fold"]), int(bank["seed"]), int(bank["inner_fold"])
        key = (fold, seed)
        outer_train = sorted(set(map(str, bank["train_subjects"])) | set(map(str, bank["held_subjects"])), key=int)
        if key not in run_cache:
            run_cache[key] = filtered_run(fold, seed, outer_train, mi_positions)
            materialized[f"{fold}_{seed}"] = {
                "subject_count": int(run_cache[key].meta.subject.nunique()),
                "row_count": len(run_cache[key].meta),
                "subjects_hash": hashlib.sha256(("\n".join(outer_train) + "\n").encode()).hexdigest(),
            }
        run_data = run_cache[key]
        if set(run_data.meta.subject.astype(str)) != set(outer_train):
            raise RuntimeError("DATA_SCOPE_VIOLATION: run cache subject mismatch")
        train_pos = positions(run_data.meta, bank["train_subjects"])
        held_pos = positions(run_data.meta, bank["held_subjects"])
        geometry = CF.fit_geometry(run_data, train_pos)
        model = fit_model(run_data, bases[key], train_pos, f"dda-a-inner-{inner}", device)
        clean_logits, _, _ = DDA.forward(model, run_data.h[held_pos], run_data.q[held_pos], device)
        deltas = np.asarray([pair["delta_q"] for pair in bank["pairs"]], dtype=np.float32)
        shifted = DDA.full_shift_logits(model, run_data.h[held_pos], run_data.q[held_pos], deltas, device)
        held_meta = run_data.meta.iloc[held_pos].reset_index(drop=True)
        info = {"fold": fold, "seed": seed, "inner_fold": inner, "run": f"{fold}_{seed}", "kind": "real_cf"}
        real_rows.extend(metric_rows(clean_logits, shifted, held_meta.label.to_numpy(dtype=np.int64), held_meta.subject.to_numpy(dtype=str), info))
        for draw in range(RANDOM_DRAWS):
            random_delta = DDA.matched_random_offsets(
                deltas, geometry, run_data.art,
                DDA.stable_seed(DDA.IMPLEMENTATION_ID, "dda-a-random", fold, seed, inner, draw),
            )
            random_shifted = DDA.full_shift_logits(model, run_data.h[held_pos], run_data.q[held_pos], random_delta, device)
            random_info = {"fold": fold, "seed": seed, "inner_fold": inner, "run": f"{fold}_{seed}", "kind": "matched_random", "draw": draw}
            random_rows.extend(metric_rows(clean_logits, random_shifted, held_meta.label.to_numpy(dtype=np.int64), held_meta.subject.to_numpy(dtype=str), random_info))
        print(f"[CF rescue/harm] bank={bank_index + 1}/{len(stress['banks'])} fold={fold} seed={seed} inner={inner}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    real = pd.DataFrame(real_rows)
    random_frame = pd.DataFrame(random_rows)
    keys = ["fold", "seed", "inner_fold", "run", "subject"]
    metrics = [
        "rescue_rate", "harm_rate", "net_rescue", "stable_correct_rate", "stable_wrong_rate",
        "true_margin_change", "correct_probability_change", "prediction_flip_rate", "ce_change",
        "ba_change", "confidence_change",
    ]
    random_mean = random_frame.groupby(keys, as_index=False)[metrics].mean().rename(columns={metric: f"random_{metric}" for metric in metrics})
    merged = real.merge(random_mean, on=keys, how="left", validate="one_to_one")
    for metric in metrics:
        merged[f"specific_{metric}"] = merged[metric] - merged[f"random_{metric}"]
    write_frame(RESULTS / "CF_RESCUE_HARM_SUBJECT.csv", merged)
    write_frame(RESULTS / "CF_RESCUE_HARM_RANDOM.csv", random_frame)
    summaries: dict[str, Any] = {}
    for column in [*metrics, *(f"specific_{metric}" for metric in metrics)]:
        bootstrap = hierarchical_bootstrap(merged, column, stable_seed(IMPLEMENTATION_ID, "bootstrap", column))
        summaries[column] = {
            "mean": float(merged[column].mean()),
            "CI95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        }
    rescue = summaries["rescue_rate"]["mean"]
    harm = summaries["harm_rate"]["mean"]
    if summaries["specific_net_rescue"]["CI95"][0] > 0:
        case = "CASE_3_CF_RESCUE_EXCEEDS_RANDOM"
    elif summaries["specific_net_rescue"]["CI95"][1] < 0:
        case = "CASE_2_CF_HARM_EXCEEDS_RESCUE_RELATIVE_TO_RANDOM"
    elif abs(rescue - harm) <= 0.002:
        case = "CASE_1_RESCUE_APPROX_HARM"
    else:
        case = "CASE_4_CF_NOT_DISTINGUISHABLE_FROM_MATCHED_RANDOM"
    result = {
        "status": "CF_RESCUE_HARM_SUPPORTING_COMPLETE",
        "case": case,
        "frozen_dda_a_status": "DDA_A_FAIL",
        "dda_a_conclusion_changed": False,
        "authorization_gate": False,
        "agdi_authorization_effect": "NONE",
        "n_real_subject_bank_rows": len(real), "n_random_subject_bank_rows": len(random_frame),
        "random_draws": RANDOM_DRAWS, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "summaries": summaries,
        "data_scope": {
            "loader": "parquet subject filters plus anonymous MI ordinal index plus npy mmap slicing",
            "all_54_subject_manifest_materialized": False,
            "all_54_subject_h0_materialized": False,
            "development_validation_rows_materialized": False,
            "outer_test_rows_materialized": False,
            "outer_labels_materialized": False,
            "only_frozen_outer_train_subjects_per_fold": True,
            "runs": materialized,
        },
        "outer_test_used": False,
    }
    write_json(RESULTS / "CF_RESCUE_HARM_RESULT.json", result)
    write_json(OUT / "protocol" / "CF_DATA_SCOPE_AUDIT.json", result["data_scope"])
    print(json.dumps(clean({key: value for key, value in result.items() if key != "summaries"}), indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    started = time.time()
    run(device)
    print(f"completed elapsed_seconds={time.time() - started:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
