#!/usr/bin/env python3
"""Frozen, same-checkpoint 48-channel rank-matched SIRE intervention.

Run --stage prepare, --stage replay, then --stage intervene, in that order.
The replay gate is deliberately hard: no projector outcome is evaluated until
all 60 original Full checkpoints reproduce the official task estimates.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/persist_eeg_sire_rankmatched_scale_collapse_v1"
PROTOCOL = EXP / "protocol"
OUT = EXP / "outputs"
RUNTIME = EXP / "runtime"
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
FOLDS = range(5)
SEEDS = range(3)
EXPECTED_PARAMS = {"OpenBMI_MI": 47978, "OpenBMI_ERP": 47978,
                   "OpenBMI_SSVEP": 48108, "WBCIC_MI": 47786}
MASTER_SEED = 271828
J = 100


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


peeh = imported("rank_control_peeh", ROOT / "experiments/persist_eeg_litebn_ablation_v2_seed0/code/run_peeh_bridge.py")
csgd = imported("rank_control_csgd", ROOT / "experiments/persist_eeg_litebn_tfformer_csgd_v1/code/run_csgd.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    pd.DataFrame(rows).to_csv(temp, index=False)
    os.replace(temp, path)


def scale_projector():
    return np.kron(np.ones((3, 3), np.float32) / 3, np.eye(16, dtype=np.float32))


def projector_error(p):
    return (float(np.max(np.abs(p - p.T))),
            float(np.max(np.abs(p @ p - p))), int(np.linalg.matrix_rank(p)))


def prepare():
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = peeh.load_runtime()
    source = []
    for task in TASKS:
        for fold in FOLDS:
            norm = peeh.normalizer_path(task, fold)
            if not norm.is_file():
                raise FileNotFoundError(norm)
            for seed in SEEDS:
                ckpt = peeh.litebn_path(task, fold, seed)
                if not ckpt.is_file():
                    raise FileNotFoundError(ckpt)
                model = runtime.base.build_model("LiteBN_BASELINE", task)
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
                model.load_state_dict(state, strict=True)
                count = sum(p.numel() for p in model.parameters())
                if count != EXPECTED_PARAMS[task]:
                    raise RuntimeError(f"wrong official SIRE parameter count: {task} f{fold} s{seed}: {count}")
                source.append({"task": task, "fold": fold, "seed": seed,
                               "checkpoint_path": str(ckpt), "checkpoint_sha256": sha(ckpt),
                               "normalizer_path": str(norm), "normalizer_sha256": sha(norm),
                               "parameter_count": count,
                               "source_experiment": "official final LiteBN/SIRE reference in carrier or openbmi task-generality runtime"})
                del model, state
    if len(source) != 60:
        raise RuntimeError("not all 60 Full checkpoints audited")
    write_csv(PROTOCOL / "CHECKPOINT_SOURCE_AUDIT.csv", source)

    pscale = scale_projector()
    sym, idem, rank = projector_error(pscale)
    if sym >= 1e-7 or idem >= 1e-7 or rank != 16:
        raise RuntimeError("scale projector algebra failed")
    rng = np.random.default_rng(924781)
    z = rng.standard_normal((48, 7)).astype(np.float32)
    a = np.concatenate([z.reshape(3, 16, 7).mean(axis=0)] * 3, axis=0)
    synthetic = float(np.max(np.abs(a - pscale @ z)))
    if synthetic >= 1e-6:
        raise RuntimeError("B2 exact collapse numerical test failed")
    write_json(OUT / "PROJECTION_NUMERICAL_AUDIT.json",
               {"scale_symmetry_error": sym, "scale_idempotence_error": idem,
                "scale_rank": rank, "synthetic_B2_max_abs_error": synthetic,
                "insertion": "after branch dropout, before first depthwise temporal backend block"})

    generator = np.random.default_rng(MASTER_SEED)
    matrices = []
    manifest = []
    overlap = []
    for j in range(J):
        seed = int(generator.integers(0, 2**63 - 1))
        g = np.random.default_rng(seed).standard_normal((48, 16))
        q, _ = np.linalg.qr(g, mode="reduced")
        p = (q @ q.T).astype(np.float32)
        sym, idem, rank = projector_error(p)
        orth = float(np.max(np.abs(q.T @ q - np.eye(16))))
        if sym >= 1e-7 or idem >= 1e-6 or orth >= 1e-10 or rank != 16:
            raise RuntimeError(f"random projector {j} algebra failed")
        matrices.append(p)
        manifest.append({"projector_id": j, "seed": seed, "rank": rank,
                         "matrix_sha256": hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest(),
                         "idempotence_error": idem, "symmetry_error": sym,
                         "orthonormality_error": orth})
        singular = np.linalg.svd(q.T @ np.kron(np.ones((3, 1)) / np.sqrt(3), np.eye(16)), compute_uv=False)
        angles = np.arccos(np.clip(singular, -1, 1))
        overlap.append({"projector_id": j, "overlap": float(np.trace(pscale @ p) / 16),
                        "principal_angles_rad": json.dumps(angles.tolist()),
                        "mean_angle_rad": float(angles.mean()),
                        "max_angle_rad": float(angles.max())})
    npz = PROTOCOL / "random_projectors.npz"
    with npz.with_suffix(".npz.part").open("wb") as f:
        np.savez_compressed(f, projectors=np.stack(matrices), scale=pscale,
                            identity=np.eye(48, dtype=np.float32), master_seed=np.int64(MASTER_SEED))
    os.replace(npz.with_suffix(".npz.part"), npz)
    write_csv(PROTOCOL / "RANDOM_PROJECTOR_MANIFEST.csv", manifest)
    write_csv(OUT / "PROJECTOR_OVERLAP_SANITY.csv", overlap)
    write_json(PROTOCOL / "PROJECTION_FREEZE.json",
               {"status": "FROZEN_BEFORE_OUTCOME_EVALUATION", "master_seed": MASTER_SEED,
                "projector_count": J, "arrays_sha256": sha(npz),
                "manifest_sha256": sha(PROTOCOL / "RANDOM_PROJECTOR_MANIFEST.csv"),
                "checkpoint_audit_sha256": sha(PROTOCOL / "CHECKPOINT_SOURCE_AUDIT.csv")})
    write_json(PROTOCOL / "PROTOCOL.json",
               {"name": "same-checkpoint rank-matched frozen ScaleCollapse control",
                "tasks": TASKS, "folds": list(FOLDS), "seeds": list(SEEDS),
                "random_projectors": J, "rank": 16, "bootstrap_subject_draws": 20000,
                "selection": "none", "neural_training": False,
                "scope": {"OpenBMI": "14-subject internal-heldout diagnostic",
                          "WBCIC": "10-subject true-outer diagnostic"},
                "insertion": "48-channel concatenated branch output after dropout, before backend",
                "previous_selector": "unchanged; not reused as architecture-ranking objective"})
    print("PREPARE_COMPLETE 60 checkpoints 100 projectors", flush=True)


def checked_sources():
    freeze = json.loads((PROTOCOL / "PROJECTION_FREEZE.json").read_text())
    for name, key in (("CHECKPOINT_SOURCE_AUDIT.csv", "checkpoint_audit_sha256"),
                      ("RANDOM_PROJECTOR_MANIFEST.csv", "manifest_sha256"),
                      ("random_projectors.npz", "arrays_sha256")):
        if sha(PROTOCOL / name) != freeze[key]:
            raise RuntimeError(f"frozen source drift: {name}")
    frame = pd.read_csv(PROTOCOL / "CHECKPOINT_SOURCE_AUDIT.csv")
    if len(frame) != 60:
        raise RuntimeError("checkpoint audit incomplete")
    return frame


def bundle_for(runtime, task):
    subjects, sessions = csgd.subjects_and_sessions(task)
    if task == "WBCIC_MI":
        bundle, _ = csgd.build_wbcic_outer_bundle(runtime)
    else:
        bundle = runtime.base.build_bundle(task, subjects)
    return bundle, subjects, sessions


def model_for(runtime, task, ckpt, device):
    model = runtime.base.build_model("LiteBN_BASELINE", task)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=True)
    model = model.to(device).eval()
    if any(layer.training for layer in model.modules()):
        raise RuntimeError("eval-mode lock failed")
    if any(p.requires_grad for p in model.parameters()):
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def original_eval(runtime, model, bundle, raw, subjects, sessions, mean, std):
    rows = []
    with torch.inference_mode():
        for subject in subjects:
            for session in sessions:
                ids = bundle.indices([subject], [session])
                if not len(ids):
                    raise RuntimeError(f"empty evaluation cell: {bundle.task} {subject} S{session}")
                logits = []
                for start in range(0, len(ids), 64):
                    x, _ = raw.batch(ids[start:start + 64], mean, std)
                    logits.append(model(x)[0].float().cpu().numpy())
                metrics = runtime.base.classification_metrics(bundle.labels(ids), np.concatenate(logits))
                rows.append({"subject_id": str(subject), "session": f"S{session}",
                             "BA": metrics["BA"], "macro_F1": metrics["macro_F1"]})
    return rows


def historical_summary():
    path = ROOT / "experiments/persist_eeg_litebn_ablation_v2_multiseed_v1/outputs/MULTISEED_TASK_SUMMARY.csv"
    frame = pd.read_csv(path)
    return frame[frame.variant == "OFFICIAL_FINAL_LITEBN_REFERENCE"].set_index("task")


def aggregate_full(rows):
    frame = pd.DataFrame(rows)
    session = frame.groupby(["task", "subject_id", "session"], as_index=False).agg(
        BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), repeats=("fold", "count"))
    if not (session.repeats == 15).all():
        raise RuntimeError("Full replay lacks 15 fold-seed repetitions per cell")
    result = {}
    for task, part in session.groupby("task"):
        subjects = []
        for _, one in part.groupby("subject_id"):
            m = one.set_index("session")
            subjects.append({"BA": float(m.loc["S2", "BA"]),
                             "macro_F1": float(m.loc["S2", "macro_F1"]),
                             "WS_BA": float(m.BA.min())})
        result[task] = pd.DataFrame(subjects).mean().to_dict()
    return result


def replay():
    audit = checked_sources()
    runtime = peeh.load_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    historical = historical_summary()
    for task in TASKS:
        bundle, subjects, sessions = bundle_for(runtime, task)
        raw = runtime.base.RawGPUCache(bundle, device)
        for record in audit[audit.task == task].itertuples(index=False):
            norm = Path(record.normalizer_path)
            ckpt = Path(record.checkpoint_path)
            if sha(norm) != record.normalizer_sha256 or sha(ckpt) != record.checkpoint_sha256:
                raise RuntimeError("checkpoint/normalizer hash drift")
            mean, std, _ = runtime.base.load_tensor_pair(norm)
            model = model_for(runtime, task, ckpt, device)
            current = original_eval(runtime, model, bundle, raw, subjects, sessions, mean, std)
            for row in current:
                row.update({"task": task, "fold": int(record.fold), "seed": int(record.seed),
                            "checkpoint_sha256": record.checkpoint_sha256})
            rows.extend(current)
            print(f"FULL_REPLAY {task} f{record.fold} s{record.seed}", flush=True)
            del model
            torch.cuda.empty_cache()
        del raw, bundle
        gc.collect(); torch.cuda.empty_cache()
    write_csv(OUT / "FULL_REPLAY_SESSION_CELLS.csv", rows)
    actual = aggregate_full(rows)
    comparison = []
    for task in TASKS:
        for metric, column in (("BA", "future_BA"), ("macro_F1", "future_macro_F1"), ("WS_BA", "WS_BA")):
            observed = float(actual[task][metric])
            expected = float(historical.loc[task, column])
            difference = abs(observed - expected)
            comparison.append({"task": task, "metric": metric, "replayed": observed,
                               "official_artifact": expected, "abs_difference": difference,
                               "pass": difference < 1e-8})
    write_csv(OUT / "FULL_REPLAY_AUDIT.csv", comparison)
    if not all(row["pass"] for row in comparison):
        raise RuntimeError("Full replay failed official artifact alignment; intervention prohibited")
    print("FULL_REPLAY_GATE_PASS", flush=True)


def branch_output(model, x):
    """Exact historical branch forward through post-pool dropout."""
    value = x.unsqueeze(1)
    branches = []
    for t, tn, s, sn in zip(model.temporal, model.temporal_norm, model.spatial, model.spatial_norm):
        branch = F.elu(tn(t(value)))
        branch = F.elu(sn(s(branch)))
        branch = F.avg_pool2d(branch, (1, 4))
        branches.append(F.dropout(branch, .20, model.training))
    return torch.cat(branches, dim=1)


def projected_logits(model, z, matrices):
    """Fixed channel projection at the audited insertion, then untouched backend."""
    n, batch = len(matrices), len(z)
    v = torch.einsum("jcd,bdht->jbcht", matrices, z).reshape(n * batch, *z.shape[1:])
    v = F.dropout(F.avg_pool2d(F.elu(model.norm1(model.point1(model.depth1(v)))), (1, 2)), .15, model.training)
    v = F.dropout(F.avg_pool2d(F.elu(model.norm2(model.point2(model.depth2(v)))), (1, 2)), .15, model.training)
    h = model.drop(model.embedding(model.pool(v).flatten(1)))
    return model.head(h).reshape(n, batch, -1)


def intervention():
    audit = checked_sources()
    replay_audit = pd.read_csv(OUT / "FULL_REPLAY_AUDIT.csv")
    if len(replay_audit) != 12 or not replay_audit["pass"].all():
        raise RuntimeError("Full replay gate absent or failed")
    with np.load(PROTOCOL / "random_projectors.npz", allow_pickle=False) as z:
        matrices_np = np.concatenate((z["identity"][None], z["scale"][None], z["projectors"]))
    if matrices_np.shape != (102, 48, 48):
        raise RuntimeError("frozen projection array schema mismatch")
    runtime = peeh.load_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrices = torch.as_tensor(matrices_np, device=device)
    identity_max = 0.0
    state_rows = []
    for task in TASKS:
        bundle, subjects, sessions = bundle_for(runtime, task)
        raw = runtime.base.RawGPUCache(bundle, device)
        for record in audit[audit.task == task].itertuples(index=False):
            output_path = RUNTIME / "cells" / task / f"fold{record.fold}_seed{record.seed}.csv"
            if output_path.is_file():
                previous = pd.read_csv(output_path)
                if len(previous) == len(subjects) * len(sessions) * 102:
                    print(f"INTERVENTION_SKIP {task} f{record.fold} s{record.seed}", flush=True)
                    continue
                raise RuntimeError(f"incomplete cell file {output_path}")
            if sha(record.checkpoint_path) != record.checkpoint_sha256 or sha(record.normalizer_path) != record.normalizer_sha256:
                raise RuntimeError("checkpoint/normalizer changed after replay")
            mean, std, _ = runtime.base.load_tensor_pair(Path(record.normalizer_path))
            model = model_for(runtime, task, Path(record.checkpoint_path), device)
            rows = []
            with torch.inference_mode():
                for subject in subjects:
                    for session in sessions:
                        ids = bundle.indices([subject], [session])
                        labels = bundle.labels(ids)
                        predictions = [[] for _ in range(102)]
                        for start in range(0, len(ids), 32):
                            x, _ = raw.batch(ids[start:start + 32], mean, std)
                            z = branch_output(model, x)
                            logits = projected_logits(model, z, matrices[:2])
                            original = model(x)[0]
                            err = float(torch.max(torch.abs(logits[0] - original)))
                            identity_max = max(identity_max, err)
                            if err >= 1e-5:
                                raise RuntimeError(f"identity wrapper mismatch {task} f{record.fold} s{record.seed}: {err}")
                            predictions[0].append(logits[0].argmax(-1).cpu().numpy())
                            predictions[1].append(logits[1].argmax(-1).cpu().numpy())
                            # Bound peak GPU memory while sharing the frozen branch computation.
                            for lo in range(2, 102, 20):
                                logits = projected_logits(model, z, matrices[lo:lo + 20])
                                for k, pred in enumerate(logits.argmax(-1).cpu().numpy()):
                                    predictions[lo + k].append(pred)
                        for j, chunks in enumerate(predictions):
                            pred = np.concatenate(chunks)
                            metrics = runtime.base.classification_metrics(labels, np.eye(int(model.head.out_features))[pred])
                            rows.append({"task": task, "fold": int(record.fold), "seed": int(record.seed),
                                         "subject_id": str(subject), "session": f"S{session}",
                                         "condition": "Intact" if j == 0 else ("ScaleCollapse-fixed" if j == 1 else "RandomRank16"),
                                         "projector_id": -1 if j < 2 else j - 2,
                                         "BA": metrics["BA"], "macro_F1": metrics["macro_F1"],
                                         "trials": len(labels)})
            write_csv(output_path, rows)
            print(f"INTERVENTION_COMPLETE {task} f{record.fold} s{record.seed} rows={len(rows)}", flush=True)
            del model
            gc.collect(); torch.cuda.empty_cache()
        del raw, bundle
        gc.collect(); torch.cuda.empty_cache()
    write_json(OUT / "IDENTITY_WRAPPER_AUDIT.json", {"max_abs_logit_difference": identity_max,
                                                     "threshold": 1e-5, "pass": identity_max < 1e-5,
                                                     "scope": "resumed process only; run --stage identity for complete 60-cell audit"})


def identity_audit():
    """Independent full-data identity check, including cells from earlier runs."""
    audit = checked_sources()
    runtime = peeh.load_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    identity = torch.eye(48, device=device, dtype=torch.float32)[None]
    rows = []
    for task in TASKS:
        bundle, subjects, sessions = bundle_for(runtime, task)
        raw = runtime.base.RawGPUCache(bundle, device)
        all_ids = np.arange(len(bundle.rows), dtype=np.int64)
        for record in audit[audit.task == task].itertuples(index=False):
            mean, std, _ = runtime.base.load_tensor_pair(Path(record.normalizer_path))
            model = model_for(runtime, task, Path(record.checkpoint_path), device)
            max_error = 0.0
            with torch.inference_mode():
                for start in range(0, len(all_ids), 32):
                    x, _ = raw.batch(all_ids[start:start + 32], mean, std)
                    z = branch_output(model, x)
                    wrapper = projected_logits(model, z, identity)[0]
                    original = model(x)[0]
                    max_error = max(max_error, float((wrapper - original).abs().max()))
            if max_error >= 1e-5:
                raise RuntimeError(f"identity audit failed {task} f{record.fold} s{record.seed}: {max_error}")
            rows.append({"task": task, "fold": int(record.fold), "seed": int(record.seed),
                         "max_abs_logit_difference": max_error, "pass": True})
            print(f"IDENTITY_AUDIT {task} f{record.fold} s{record.seed} max={max_error:.9g}", flush=True)
            del model
            torch.cuda.empty_cache()
        del raw, bundle
        gc.collect(); torch.cuda.empty_cache()
    write_csv(OUT / "IDENTITY_WRAPPER_CELLS.csv", rows)
    write_json(OUT / "IDENTITY_WRAPPER_AUDIT.json",
               {"max_abs_logit_difference": max(row["max_abs_logit_difference"] for row in rows),
                "threshold": 1e-5, "pass": len(rows) == 60,
                "scope": "all 60 frozen checkpoints, all evaluation trials"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("prepare", "replay", "intervene", "identity"))
    args = parser.parse_args()
    {"prepare": prepare, "replay": replay, "intervene": intervention, "identity": identity_audit}[args.stage]()


if __name__ == "__main__":
    main()
