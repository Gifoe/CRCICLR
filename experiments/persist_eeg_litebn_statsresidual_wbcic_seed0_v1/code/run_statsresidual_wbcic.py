"""Run the fixed LiteBN-StatsResidual WBCIC-MI seed-0 experiment."""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from aggregate_statsresidual import decide, metric_summary, paired_bootstrap, signs
from litebn_statsresidual import (
    LiteBNStatsResidual,
    exact_state_audit,
    split_parameter_buffer_schema,
    trainable_parameter_count,
)


TASK = "WBCIC_MI"
DATASET = "WBCIC"
SEED = 0
TIE_TOL = 1e-12
BETA = 1e-3
LR = 1e-4
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 30
PATIENCE = 5
CLIP = 5.0
BATCH_SIZE = 512
PRECOMPUTE_BATCH = 256
METHOD = "LiteBN-StatsResidual"
VERSION = "statsres-wbcic-seed0-v1-historical-64ch"
METRICS = ("BA", "macro_F1", "accuracy")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([clean(row) for row in rows])
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def score(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "BA": float(balanced_accuracy_score(labels, predictions)),
        "macro_F1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def subject_equal_metrics(labels: np.ndarray, predictions: np.ndarray, subjects: list[str]) -> dict[str, float]:
    rows = []
    subject_array = np.asarray(subjects, dtype=object)
    for subject in sorted(set(subjects)):
        mask = subject_array == subject
        rows.append(score(labels[mask], predictions[mask]))
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}


def state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


class Experiment:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = args.repo.resolve()
        self.recovered = args.recovered.resolve()
        self.snapshot = self.recovered / "snapshot"
        self.exp = self.repo / "experiments" / "persist_eeg_litebn_statsresidual_wbcic_seed0_v1"
        self.out = self.exp / "outputs"
        self.protocol = self.exp / "protocol"
        self.runtime = args.runtime.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.protocol.mkdir(parents=True, exist_ok=True)
        self.runtime.mkdir(parents=True, exist_ok=True)

        os.environ.update(
            R2EEG_REPO=str(self.repo),
            TASK_GENERALITY_REPO=str(self.repo),
            FINAL_CONFIRM_REPO=str(self.repo),
            PERSIST_OPENBMI_CACHE=str(args.cache.resolve() / "openbmi" / "openbmi"),
            PERSIST_WBCIC_CACHE=str(args.cache.resolve() / "wbcic" / "wbcic_epochs"),
        )
        self.carrier = self.snapshot / "persist_eeg_carrier_5fold_multiseed_stability_v1"
        self.final = self.snapshot / "persist_eeg_final_heldout_confirmation_v1"
        sys.path[:0] = [
            str(self.carrier / "code"),
            str(self.final / "code"),
            str(self.repo / "experiments" / "persist_eeg_litebn_lighttail_seed0_v1" / "code"),
        ]
        self.historical = importlib.import_module("train_grid")
        self.v1 = self.historical.v1
        self.final_loader = importlib.import_module("load_final_carriers")
        self.legacy = importlib.import_module("legacy_init")

        self.split_path = self.carrier / "protocol" / "FIVEFOLD_SPLIT.json"
        self.logs_path = self.carrier / "outputs" / "TRAINING_LOGS.json"
        self.manifest_meta_path = self.carrier / "protocol" / "MANIFEST_HASHES.json"
        self.outer_reference_path = self.carrier / "outputs" / "SUBJECT_SEED_RESULTS.csv"
        self.heldout_manifest_path = self.final / "protocol" / "FINAL_HOLDOUT_MANIFEST.json"
        self.heldout_reference_path = self.final / "outputs" / "REPLICATE_SUBJECT_RESULTS.csv"
        for path in (
            self.split_path, self.logs_path, self.manifest_meta_path, self.outer_reference_path,
            self.heldout_manifest_path, self.heldout_reference_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.split = json.loads(self.split_path.read_text(encoding="utf-8"))
        self.logs = json.loads(self.logs_path.read_text(encoding="utf-8"))
        self.manifest_meta = json.loads(self.manifest_meta_path.read_text(encoding="utf-8"))
        self.heldout = json.loads(self.heldout_manifest_path.read_text(encoding="utf-8"))
        self.historical.validate_split(self.split["search_subjects"], self.split["folds"])
        if set(self.split["search_subjects"][DATASET]) & set(self.heldout[DATASET]["subject_ids"]):
            raise RuntimeError("STATSRES_PROTOCOL_FAIL heldout/search overlap")
        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this frozen protocol run")
        torch.set_num_threads(4)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        set_seed(SEED)
        self.bundle = self.v1.load_bundle(DATASET, self.split["search_subjects"][DATASET])
        self.source_files = [
            Path(__file__), Path(__file__).with_name("litebn_statsresidual.py"),
            Path(__file__).with_name("aggregate_statsresidual.py"), Path(self.historical.__file__),
            Path(self.v1.__file__), Path(self.final_loader.__file__), Path(self.legacy.__file__),
            self.split_path, self.logs_path, self.manifest_meta_path, self.heldout_manifest_path,
        ]
        self.source_hashes = {str(path): sha_file(path) for path in self.source_files}
        self.invariant = hashlib.sha256(json.dumps(self.source_hashes, sort_keys=True).encode()).hexdigest()

    def fold(self, fold_id: int) -> dict[str, Any]:
        return self.split["folds"][DATASET][fold_id]

    def record(self, fold_id: int) -> dict[str, Any]:
        rows = [
            row for row in self.logs
            if row["dataset"] == DATASET and row["model"] == "LiteBN"
            and int(row["fold"]) == fold_id and int(row["seed"]) == SEED
        ]
        if len(rows) != 1:
            raise RuntimeError(f"STATSRES_PROTOCOL_FAIL reference record count fold {fold_id}: {len(rows)}")
        return rows[0]

    def checkpoint(self, fold_id: int) -> Path:
        return self.recovered / "checkpoints" / TASK / f"fold{fold_id}_seed0" / "selected_best.pt"

    def normalizer(self, fold_id: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        fold = self.fold(fold_id)
        mean, std, metadata = self.v1.normalizer(self.bundle, fold["inner_train_subjects"])
        wanted = [row["normalizer"] for row in self.manifest_meta if row["dataset"] == DATASET and int(row["fold"]) == fold_id]
        if len(wanted) != 1 or metadata != wanted[0]:
            raise RuntimeError(f"STATSRES_PROTOCOL_FAIL normalizer mismatch fold {fold_id}")
        return mean, std, metadata

    def manifest_path(self, fold_id: int) -> Path:
        return self.args.historical_runtime.resolve() / f"wbcic_fold{fold_id}" / "manifest_runtime" / "episode_manifests" / f"wbcic_fold{fold_id}.json"

    def load_base(self, fold_id: int) -> torch.nn.Module:
        record, checkpoint = self.record(fold_id), self.checkpoint(fold_id)
        if not checkpoint.is_file() or sha_file(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError(f"STATSRES_PROTOCOL_FAIL checkpoint provenance mismatch fold {fold_id}")

        def constructor(name: str, _unused: int) -> torch.nn.Module:
            return self.historical.constructor(name, 58)

        base, _audit = self.legacy.construct_exact(
            constructor, self.historical.set_seed, checkpoint, record["init_sha256"], self.device
        )
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        base.load_state_dict(state, strict=True)
        base.eval()
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        shapes = {name: tuple(value.shape) for name, value in base.state_dict().items()}
        required = {
            "depth1.weight": (48, 1, 1, 15),
            "point1.weight": (64, 48, 1, 1),
            "depth2.weight": (64, 1, 1, 31),
            "point2.weight": (64, 64, 1, 1),
            "head.weight": (2, 64),
        }
        if any(shapes.get(name) != shape for name, shape in required.items()):
            raise RuntimeError(f"STATSRES_PROTOCOL_FAIL non-historical LiteBN architecture fold {fold_id}")
        return base

    def normalized_batch(self, indices: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
        return self.v1.prepare(self.bundle, indices, mean, std, self.device)

    def precompute_indices(
        self, model: LiteBNStatsResidual, indices: np.ndarray, mean: np.ndarray, std: np.ndarray,
    ) -> dict[str, Any]:
        logits, stats = [], []
        for start in range(0, len(indices), PRECOMPUTE_BATCH):
            x = self.normalized_batch(indices[start:start + PRECOMPUTE_BATCH], mean, std)
            z0, residual_features = model.base_forward_and_stats(x)
            logits.append(z0.float())
            stats.append(residual_features.float())
        labels = self.bundle.labels(indices)
        subjects = [self.bundle.search_rows[int(index)].subject for index in indices]
        return {
            "indices": indices.copy(),
            "z0": torch.cat(logits),
            "stats": torch.cat(stats),
            "labels": torch.as_tensor(labels, device=self.device, dtype=torch.long),
            "labels_numpy": labels,
            "subjects": subjects,
        }

    def cache_audit(
        self, fold_id: int, split_name: str, model: LiteBNStatsResidual, cache: dict[str, Any],
        mean: np.ndarray, std: np.ndarray,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(81_001 + fold_id + (10_000 if split_name == "inner_val" else 0))
        positions = np.sort(rng.choice(len(cache["indices"]), size=min(17, len(cache["indices"])), replace=False))
        indices = cache["indices"][positions]
        x = self.normalized_batch(indices, mean, std)
        z0, stats = model.base_forward_and_stats(x)
        z_diff = float((z0.float() - cache["z0"][positions]).abs().max())
        stats_diff = float((stats.float() - cache["stats"][positions]).abs().max())
        label_mismatch = int((torch.as_tensor(self.bundle.labels(indices), device=self.device) != cache["labels"][positions]).sum())
        # The same FP32 convolution can differ by a few ulps when the audit
        # subset changes GEMM batch shape; 1e-6 is a tight numerical replay.
        passed = z_diff <= 1e-6 and stats_diff <= 1e-6 and label_mismatch == 0
        return {
            "task": TASK, "seed": SEED, "fold": fold_id, "cache_split": split_name,
            "audited_samples": len(positions), "max_abs_base_logit_difference": z_diff,
            "max_abs_stats_difference": stats_diff, "label_mismatch_count": label_mismatch,
            "absolute_tolerance": 1e-6,
            "status": "PASS" if passed else "FAIL",
        }

    def epoch_metrics(self, head: torch.nn.Module, cache: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
        with torch.no_grad():
            residual_logits = head(cache["stats"])
            final_logits = cache["z0"] + residual_logits
            predictions = final_logits.argmax(dim=1).cpu().numpy()
            base_predictions = cache["z0"].argmax(dim=1).cpu().numpy()
            residual_norms = torch.linalg.vector_norm(residual_logits, dim=1)
            cosine = F.cosine_similarity(cache["z0"], final_logits, dim=1)
        metrics = subject_equal_metrics(cache["labels_numpy"], predictions, cache["subjects"])
        diagnostics = {
            "residual_logit_norm_mean": float(residual_norms.mean()),
            "residual_logit_norm_max": float(residual_norms.max()),
            "base_final_logit_cosine_mean": float(cosine.mean()),
            "changed_prediction_percentage": 100.0 * float(np.mean(predictions != base_predictions)),
        }
        return metrics, diagnostics

    def run_fold(self, fold_id: int) -> dict[str, Any]:
        cell = self.runtime / f"fold{fold_id}"
        cell.mkdir(parents=True, exist_ok=True)
        result_path = cell / "result.json"
        if result_path.is_file():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved.get("invariant") == self.invariant:
                print(f"FOLD {fold_id} REUSED", flush=True)
                return saved

        fold, record, checkpoint = self.fold(fold_id), self.record(fold_id), self.checkpoint(fold_id)
        mean, std, normalizer = self.normalizer(fold_id)
        manifest_path = self.manifest_path(fold_id)
        if not manifest_path.is_file() or sha_file(manifest_path) != record["manifest_sha256"]:
            raise RuntimeError(f"STATSRES_PROTOCOL_FAIL manifest provenance mismatch fold {fold_id}")
        train_subjects = list(map(str, fold["inner_train_subjects"]))
        val_subjects = list(map(str, fold["inner_val_subjects"]))
        outer_subjects = list(map(str, fold["outer_dev_subjects"]))
        heldout_subjects = list(map(str, self.heldout[DATASET]["subject_ids"]))
        sets = list(map(set, (train_subjects, val_subjects, outer_subjects, heldout_subjects)))
        if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))):
            raise RuntimeError(f"STATSRES_PROTOCOL_FAIL subject leakage fold {fold_id}")
        train_indices = self.bundle.indices(train_subjects, (0, 1))
        val_indices = self.bundle.indices(val_subjects, (2,))
        if len(train_indices) == 0 or len(val_indices) == 0:
            raise RuntimeError(f"STATSRES_PROTOCOL_FAIL empty residual split fold {fold_id}")

        base = self.load_base(fold_id)
        parameter_schema, buffer_schema = split_parameter_buffer_schema(base)
        base_parameters = sum(parameter.numel() for parameter in base.parameters())
        model = LiteBNStatsResidual(base).to(self.device)
        if trainable_parameter_count(model) != 258:
            raise RuntimeError("STATSRES_PROTOCOL_FAIL TRAINABLE_PARAMETER_COUNT != 258")
        if model.base.training or any(parameter.requires_grad for parameter in model.base.parameters()):
            raise RuntimeError("STATSRES_PROTOCOL_FAIL base not frozen/eval")
        before_state = state_cpu(model.base)

        started = time.perf_counter()
        train_cache = self.precompute_indices(model, train_indices, mean, std)
        val_cache = self.precompute_indices(model, val_indices, mean, std)
        cache_audits = [
            self.cache_audit(fold_id, "inner_train", model, train_cache, mean, std),
            self.cache_audit(fold_id, "inner_val", model, val_cache, mean, std),
        ]
        print(f"CACHE_AUDIT fold={fold_id} {cache_audits}", flush=True)
        if any(row["status"] != "PASS" for row in cache_audits):
            raise RuntimeError(f"STATSRES_PROTOCOL_FAIL feature cache audit fold {fold_id}")

        with torch.no_grad():
            epoch0_final = val_cache["z0"] + model.residual(val_cache["stats"])
        epoch0_difference = float((epoch0_final - val_cache["z0"]).abs().max())
        epoch0_prediction_mismatch = int((epoch0_final.argmax(1) != val_cache["z0"].argmax(1)).sum())
        epoch0_base = subject_equal_metrics(
            val_cache["labels_numpy"], val_cache["z0"].argmax(1).cpu().numpy(), val_cache["subjects"]
        )
        epoch0_metrics, epoch0_diagnostics = self.epoch_metrics(model.residual, val_cache)
        epoch0_audit = {
            "task": TASK, "seed": SEED, "fold": fold_id,
            "max_abs_logit_difference": epoch0_difference,
            "prediction_mismatch_count": epoch0_prediction_mismatch,
            "BA_difference": epoch0_metrics["BA"] - epoch0_base["BA"],
            "macro_F1_difference": epoch0_metrics["macro_F1"] - epoch0_base["macro_F1"],
            "accuracy_difference": epoch0_metrics["accuracy"] - epoch0_base["accuracy"],
            "status": "PASS",
        }
        if not (
            epoch0_difference <= 1e-7 and epoch0_prediction_mismatch == 0
            and all(epoch0_audit[f"{metric}_difference"] == 0.0 for metric in METRICS)
        ):
            epoch0_audit["status"] = "FAIL"
            raise RuntimeError(f"STATSRES_PROTOCOL_FAIL epoch0 replay fold {fold_id}")

        with torch.no_grad():
            train_ce0 = float(F.cross_entropy(train_cache["z0"], train_cache["labels"]))
        history: list[dict[str, Any]] = [{
            "task": TASK, "seed": SEED, "fold": fold_id, "epoch": 0,
            "inner_val_BA": epoch0_metrics["BA"], "inner_val_macro_F1": epoch0_metrics["macro_F1"],
            "inner_val_accuracy": epoch0_metrics["accuracy"], "delta_BA_vs_epoch0_pp": 0.0,
            "dW_norm": 0.0, "db_norm": 0.0, **epoch0_diagnostics,
            "training_CE": train_ce0, "regularization_term": 0.0,
            "selected_at_epoch": True,
        }]
        best_ba = epoch0_metrics["BA"]
        best_epoch = 0
        best_state = copy.deepcopy(model.residual.state_dict())
        stale = 0
        optimizer = torch.optim.AdamW(model.residual.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(10_003 + fold_id)
        shuffle_digest = hashlib.sha256()

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train(True)
            if model.base.training:
                raise RuntimeError("STATSRES_PROTOCOL_FAIL base entered train mode")
            permutation = torch.randperm(len(train_indices), generator=generator).numpy()
            shuffle_digest.update(permutation.astype(np.int64, copy=False).tobytes())
            ce_sum, reg_sum, observed = 0.0, 0.0, 0
            for start in range(0, len(permutation), BATCH_SIZE):
                positions = torch.as_tensor(permutation[start:start + BATCH_SIZE], device=self.device, dtype=torch.long)
                z0 = train_cache["z0"].index_select(0, positions)
                features = train_cache["stats"].index_select(0, positions)
                labels = train_cache["labels"].index_select(0, positions)
                optimizer.zero_grad(set_to_none=True)
                residual_logits = model.residual(features)
                ce = F.cross_entropy(z0 + residual_logits, labels)
                regularizer = BETA * (
                    model.residual.weight.square().mean() + model.residual.bias.square().mean()
                )
                loss = ce + regularizer
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite residual loss fold={fold_id} epoch={epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.residual.parameters(), CLIP)
                optimizer.step()
                batch_count = len(positions)
                ce_sum += float(ce.detach()) * batch_count
                reg_sum += float(regularizer.detach()) * batch_count
                observed += batch_count
            model.eval()
            validation, diagnostics = self.epoch_metrics(model.residual, val_cache)
            improved = validation["BA"] > best_ba + TIE_TOL
            if improved:
                best_ba, best_epoch = validation["BA"], epoch
                best_state = copy.deepcopy(model.residual.state_dict())
                stale = 0
            else:
                stale += 1
            history.append({
                "task": TASK, "seed": SEED, "fold": fold_id, "epoch": epoch,
                "inner_val_BA": validation["BA"], "inner_val_macro_F1": validation["macro_F1"],
                "inner_val_accuracy": validation["accuracy"],
                "delta_BA_vs_epoch0_pp": 100.0 * (validation["BA"] - epoch0_metrics["BA"]),
                "dW_norm": float(torch.linalg.vector_norm(model.residual.weight.detach())),
                "db_norm": float(torch.linalg.vector_norm(model.residual.bias.detach())),
                **diagnostics, "training_CE": ce_sum / observed,
                "regularization_term": reg_sum / observed, "selected_at_epoch": bool(improved),
            })
            print(
                f"[StatsResidual WBCIC f{fold_id}] epoch={epoch:02d} "
                f"valBA={validation['BA']:.6f} best={best_ba:.6f}@{best_epoch} stale={stale}",
                flush=True,
            )
            if stale >= PATIENCE:
                break

        model.residual.load_state_dict(best_state, strict=True)
        model.eval()
        selected_metrics, selected_diagnostics = self.epoch_metrics(model.residual, val_cache)
        after_state = state_cpu(model.base)
        freeze_audit = {"task": TASK, "seed": SEED, "fold": fold_id, **exact_state_audit(before_state, after_state)}
        if freeze_audit["status"] != "PASS":
            raise RuntimeError(f"STATSRES_PROTOCOL_FAIL base drift fold {fold_id}")
        selected_checkpoint = cell / "selected_statsresidual.pt"
        torch.save({
            "residual_state_dict": {name: value.detach().cpu() for name, value in best_state.items()},
            "selected_epoch": best_epoch, "invariant": self.invariant,
            "base_checkpoint_sha256": sha_file(checkpoint), "normalizer_sha256": normalizer["mean_std_sha256"],
        }, selected_checkpoint)
        manifest_row = {
            "task": TASK, "seed": SEED, "fold": fold_id,
            "inner_train_subjects": "|".join(train_subjects), "inner_val_subjects": "|".join(val_subjects),
            "outer_subjects": "|".join(outer_subjects), "internal_heldout_subjects": "|".join(heldout_subjects),
            "inner_train_trials": len(train_indices), "inner_val_trials": len(val_indices),
            "source_sessions": "0|1", "validation_sessions": "2",
            "split_sha256": sha_file(self.split_path), "historical_manifest_path": str(manifest_path),
            "historical_manifest_sha256": sha_file(manifest_path),
            "normalizer_sha256": normalizer["mean_std_sha256"],
            "shuffle_seed": 10_003 + fold_id, "shuffle_manifest_sha256": shuffle_digest.hexdigest(),
            "no_subject_overlap": True, "status": "PASS",
        }
        provenance = {
            "task": TASK, "seed": SEED, "fold": fold_id,
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha_file(checkpoint),
            "selected_epoch": int(record["selected_epoch"]), "parameter_count": base_parameters,
            "parameter_schema": parameter_schema, "buffer_schema": buffer_schema,
            "normalizer_sha256": normalizer["mean_std_sha256"],
            "manifest_path": str(manifest_path), "manifest_sha256": sha_file(manifest_path),
            "historical_record": clean(record),
        }
        result = {
            "invariant": self.invariant, "fold": fold_id, "history": history,
            "epoch0_audit": epoch0_audit, "cache_audits": cache_audits,
            "freeze_audit": freeze_audit, "manifest_audit": manifest_row, "provenance": provenance,
            "selected_epoch": best_epoch, "epoch0_BA": epoch0_metrics["BA"],
            "selected_BA": selected_metrics["BA"],
            "delta_BA_vs_epoch0_pp": 100.0 * (selected_metrics["BA"] - epoch0_metrics["BA"]),
            "selected_residual_head_norm": float(torch.sqrt(
                model.residual.weight.detach().square().sum() + model.residual.bias.detach().square().sum()
            )),
            "selected_changed_prediction_percentage": selected_diagnostics["changed_prediction_percentage"],
            "residual_checkpoint_path": str(selected_checkpoint),
            "residual_checkpoint_sha256": sha_file(selected_checkpoint),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(result_path, result)
        model.close()
        del model, base, train_cache, val_cache
        gc.collect()
        torch.cuda.empty_cache()
        print(f"FOLD {fold_id} COMPLETE selected_epoch={best_epoch}", flush=True)
        return result

    def load_selected(self, fold_id: int) -> tuple[LiteBNStatsResidual, np.ndarray, np.ndarray, dict[str, Any]]:
        result = json.loads((self.runtime / f"fold{fold_id}" / "result.json").read_text(encoding="utf-8"))
        if result["invariant"] != self.invariant:
            raise RuntimeError(f"runtime invariant mismatch fold {fold_id}")
        checkpoint = Path(result["residual_checkpoint_path"])
        if sha_file(checkpoint) != result["residual_checkpoint_sha256"]:
            raise RuntimeError(f"residual checkpoint hash mismatch fold {fold_id}")
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        model = LiteBNStatsResidual(self.load_base(fold_id)).to(self.device)
        model.residual.load_state_dict(payload["residual_state_dict"], strict=True)
        model.eval()
        mean, std, _ = self.normalizer(fold_id)
        return model, mean, std, result

    def infer_raw(self, model: LiteBNStatsResidual, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base_all, final_all = [], []
        with torch.no_grad():
            for start in range(0, len(x), 128):
                value = x[start:start + 128].astype(np.float32, copy=False)
                value = (value - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
                tensor = torch.from_numpy(np.ascontiguousarray(value)).to(self.device)
                final, base, _stats = model(tensor)
                base_all.append(base.float().cpu().numpy())
                final_all.append(final.float().cpu().numpy())
        return np.concatenate(base_all), np.concatenate(final_all)

    def historical_outer(self, fold_id: int) -> dict[str, dict[str, float]]:
        frame = pd.read_csv(self.outer_reference_path)
        frame = frame[(frame.dataset == DATASET) & (frame.fold == fold_id) & (frame.seed == SEED)]
        if len(frame) != len(self.fold(fold_id)["outer_dev_subjects"]):
            raise RuntimeError(f"outer baseline reference count mismatch fold {fold_id}")
        return {
            str(row.subject_id): {metric: float(getattr(row, f"LiteBN_{metric}")) for metric in METRICS}
            for row in frame.itertuples(index=False)
        }

    def historical_heldout(self, fold_id: int) -> dict[str, dict[str, float]]:
        frame = pd.read_csv(self.heldout_reference_path)
        frame = frame[
            (frame.dataset == DATASET) & (frame.fold == fold_id) & (frame.seed == SEED) & (frame.method == "LiteBN")
        ]
        if len(frame) != len(self.heldout[DATASET]["subject_ids"]):
            raise RuntimeError(f"heldout baseline reference count mismatch fold {fold_id}")
        return {
            str(row.subject_id): {metric: float(getattr(row, metric)) for metric in METRICS}
            for row in frame.itertuples(index=False)
        }

    def evaluate_outer(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for fold_id in range(5):
            model, mean, std, result = self.load_selected(fold_id)
            historical = self.historical_outer(fold_id)
            for subject in map(str, self.fold(fold_id)["outer_dev_subjects"]):
                indices = self.bundle.indices([subject], (2,))
                x, y = self.bundle.accessor.batch(indices), self.bundle.labels(indices)
                base_logits, final_logits = self.infer_raw(model, x, mean, std)
                base_score, candidate_score = score(y, base_logits.argmax(1)), score(y, final_logits.argmax(1))
                for metric in METRICS:
                    if abs(base_score[metric] - historical[subject][metric]) > 1e-10:
                        raise RuntimeError(f"STATSRES_PROTOCOL_FAIL outer B0 replay fold={fold_id} subject={subject}")
                row: dict[str, Any] = {
                    "task": TASK, "seed": SEED, "fold": fold_id, "subject_id": subject, "trials": len(y),
                    "selected_epoch": result["selected_epoch"],
                    "changed_prediction_percentage": 100.0 * float(np.mean(base_logits.argmax(1) != final_logits.argmax(1))),
                }
                for metric in METRICS:
                    row[f"B0_{metric}"] = base_score[metric]
                    row[f"StatsResidual_{metric}"] = candidate_score[metric]
                    row[f"delta_{metric}_pp"] = 100.0 * (candidate_score[metric] - base_score[metric])
                rows.append(row)
            model.close()
            del model
            gc.collect(); torch.cuda.empty_cache()
            print(f"OUTER FOLD {fold_id} COMPLETE", flush=True)
        write_csv(self.out / "WBCIC_STATSRES_OUTER_SUBJECT_RESULTS.csv", rows)
        return rows

    def evaluate_heldout(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        subjects = list(map(str, self.heldout[DATASET]["subject_ids"]))
        heldout_data = {subject: self.final_loader.load_eval_subject(DATASET, subject) for subject in subjects}
        for fold_id in range(5):
            model, mean, std, result = self.load_selected(fold_id)
            historical = self.historical_heldout(fold_id)
            for subject in subjects:
                x, y = heldout_data[subject]
                base_logits, final_logits = self.infer_raw(model, x, mean, std)
                base_score, candidate_score = score(y, base_logits.argmax(1)), score(y, final_logits.argmax(1))
                for metric in METRICS:
                    if abs(base_score[metric] - historical[subject][metric]) > 1e-10:
                        raise RuntimeError(f"STATSRES_PROTOCOL_FAIL heldout B0 replay fold={fold_id} subject={subject}")
                row: dict[str, Any] = {
                    "task": TASK, "seed": SEED, "fold": fold_id, "subject_id": subject, "trials": len(y),
                    "selected_epoch": result["selected_epoch"],
                    "changed_prediction_percentage": 100.0 * float(np.mean(base_logits.argmax(1) != final_logits.argmax(1))),
                }
                for metric in METRICS:
                    row[f"B0_{metric}"] = base_score[metric]
                    row[f"StatsResidual_{metric}"] = candidate_score[metric]
                    row[f"delta_{metric}_pp"] = 100.0 * (candidate_score[metric] - base_score[metric])
                rows.append(row)
            model.close()
            del model
            gc.collect(); torch.cuda.empty_cache()
            print(f"INTERNAL HELDOUT FOLD {fold_id} COMPLETE", flush=True)
        return rows

    def finalize(self, fold_results: list[dict[str, Any]], outer_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]]) -> str:
        histories = [row for result in fold_results for row in result["history"]]
        epoch0 = [result["epoch0_audit"] for result in fold_results]
        caches = [row for result in fold_results for row in result["cache_audits"]]
        freeze = [result["freeze_audit"] for result in fold_results]
        manifests = [result["manifest_audit"] for result in fold_results]
        provenance = [result["provenance"] for result in fold_results]
        inner = [{
            "task": TASK, "seed": SEED, "fold": result["fold"], "epoch0_BA": result["epoch0_BA"],
            "selected_BA": result["selected_BA"], "delta_BA_vs_epoch0_pp": result["delta_BA_vs_epoch0_pp"],
            "selected_epoch": result["selected_epoch"], "selected_residual_head_norm": result["selected_residual_head_norm"],
            "selected_changed_prediction_percentage": result["selected_changed_prediction_percentage"],
        } for result in fold_results]
        write_csv(self.out / "STATSRES_EPOCH0_EXACT_B0_REPLAY.csv", epoch0)
        write_csv(self.out / "STATSRES_BASE_FREEZE_AUDIT.csv", freeze)
        write_csv(self.out / "STATSRES_FEATURE_CACHE_AUDIT.csv", caches)
        write_csv(self.out / "STATSRES_MANIFEST_AUDIT.csv", manifests)
        write_csv(self.out / "STATSRES_INNER_HISTORY.csv", histories)
        write_csv(self.out / "STATSRES_INNER_FOLD_RESULTS.csv", inner)
        write_json(self.protocol / "REFERENCE_PROVENANCE.json", {
            "reference_choice": "historical_64_channel_LiteBN_explicitly_selected_by_user",
            "supersedes_prompt_same_linux_numeric_reference": True,
            "records": provenance,
            "source_artifact_hashes": {
                "split": sha_file(self.split_path), "training_logs": sha_file(self.logs_path),
                "manifest_metadata": sha_file(self.manifest_meta_path),
                "outer_subject_results": sha_file(self.outer_reference_path),
                "heldout_manifest": sha_file(self.heldout_manifest_path),
                "heldout_replicate_results": sha_file(self.heldout_reference_path),
            },
        })

        outer = pd.DataFrame(outer_rows)
        outer_fold = outer.groupby(["task", "seed", "fold"], as_index=False).agg({
            **{f"B0_{metric}": "mean" for metric in METRICS},
            **{f"StatsResidual_{metric}": "mean" for metric in METRICS},
            **{f"delta_{metric}_pp": "mean" for metric in METRICS},
            "changed_prediction_percentage": "mean", "selected_epoch": "first",
        })
        outer_fold.to_csv(self.out / "WBCIC_STATSRES_OUTER_FOLD_RESULTS.csv", index=False)
        outer_summary = metric_summary(outer)
        outer_subject_deltas = outer["delta_BA_pp"].to_numpy(float)
        outer_bootstrap = paired_bootstrap(outer_subject_deltas, seed=0)
        outer_signs = signs(outer_subject_deltas)
        fold_deltas = outer_fold["delta_BA_pp"].to_numpy(float)
        fold_signs = signs(fold_deltas)
        outer_summary_row = {
            "task": TASK, "seed": SEED, "subjects": len(outer), "folds": 5,
            **outer_summary,
            "positive_folds": fold_signs["positive"], "negative_folds": fold_signs["negative"], "tied_folds": fold_signs["tied"],
            "worst_fold_delta_pp": float(fold_deltas.min()),
            "positive_subjects": outer_signs["positive"], "negative_subjects": outer_signs["negative"], "tied_subjects": outer_signs["tied"],
            "bootstrap_ci95_low_pp": outer_bootstrap["ci95_low_pp"], "bootstrap_ci95_high_pp": outer_bootstrap["ci95_high_pp"],
            "mean_changed_prediction_percentage": float(np.average(outer.changed_prediction_percentage, weights=outer.trials)),
        }
        write_csv(self.out / "WBCIC_STATSRES_OUTER_SUMMARY.csv", [outer_summary_row])
        write_csv(self.out / "WBCIC_STATSRES_OUTER_BOOTSTRAP.csv", [{"task": TASK, "seed": SEED, **outer_bootstrap}])

        heldout_replicates = pd.DataFrame(heldout_rows)
        heldout_subject = heldout_replicates.groupby(["task", "seed", "subject_id"], as_index=False).agg({
            **{f"B0_{metric}": "mean" for metric in METRICS},
            **{f"StatsResidual_{metric}": "mean" for metric in METRICS},
            **{f"delta_{metric}_pp": "mean" for metric in METRICS},
            "changed_prediction_percentage": "mean", "trials": "first",
        })
        heldout_subject.to_csv(self.out / "WBCIC_STATSRES_INTERNAL_HELDOUT_SUBJECT_RESULTS.csv", index=False)
        heldout_summary = metric_summary(heldout_subject)
        heldout_deltas = heldout_subject["delta_BA_pp"].to_numpy(float)
        heldout_signs = signs(heldout_deltas)
        heldout_bootstrap = paired_bootstrap(heldout_deltas, seed=0)
        heldout_summary_row = {
            "task": TASK, "seed": SEED, "real_subjects": len(heldout_subject), "fold_replicates_per_subject": 5,
            "aggregation_rule": "mean fold replicates within real subject, then mean real subjects",
            "status": "DEVELOPMENT_MODEL_SELECTION_DATA", **heldout_summary,
            "positive_subjects": heldout_signs["positive"], "negative_subjects": heldout_signs["negative"], "tied_subjects": heldout_signs["tied"],
            "bootstrap_ci95_low_pp": heldout_bootstrap["ci95_low_pp"], "bootstrap_ci95_high_pp": heldout_bootstrap["ci95_high_pp"],
        }
        write_csv(self.out / "WBCIC_STATSRES_INTERNAL_HELDOUT_SUMMARY.csv", [heldout_summary_row])

        protocol_pass = all(row["status"] == "PASS" for row in epoch0 + caches + freeze + manifests)
        terminal, criteria = decide(
            outer_summary_row["delta_BA_pp"], fold_deltas, outer_subject_deltas,
            heldout_summary_row["delta_BA_pp"], protocol_pass,
        )
        selected_epochs = [int(result["selected_epoch"]) for result in fold_results]
        decision = {
            "terminal": terminal, "criteria": criteria, "model": METHOD, "task": TASK, "seed": SEED,
            "base_model": "EXACT_FROZEN_TRAINED_HISTORICAL_64CH_LITEBN",
            "base_model_trainable": False, "base_model_parameter_drift": 0.0,
            "residual_dim": 128, "new_trainable_parameters": 258,
            "selected_epochs": selected_epochs, "folds_selecting_epoch0": selected_epochs.count(0),
            "mean_selected_residual_head_norm": float(np.mean([result["selected_residual_head_norm"] for result in fold_results])),
            "outer": outer_summary_row, "outer_bootstrap": outer_bootstrap,
            "internal_heldout": heldout_summary_row, "internal_heldout_bootstrap": heldout_bootstrap,
            "new_sealed_test_accessed": False, "invariant": self.invariant,
        }
        write_json(self.out / "FINAL_STATSRES_DECISION.json", decision)
        write_json(self.protocol / "RUNTIME_METADATA.json", {
            "version": VERSION, "intended_branch": "codex/persist-eeg-litebn-statsresidual-wbcic-seed0-v1",
            "platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "device": str(self.device),
            "source_hashes": self.source_hashes, "invariant": self.invariant,
        })
        report = f"""# LiteBN-StatsResidual WBCIC seed0 report

MODEL = LiteBN-StatsResidual
TASK = WBCIC_MI
SEED = 0

BASE_MODEL = EXACT_FROZEN_TRAINED_B0_LITEBN
BASE_MODEL_TRAINABLE = NO
BASE_MODEL_PARAMETER_DRIFT = 0

RESIDUAL_FEATURE = CONCAT(FINAL_TEMPORAL_FEATURE_MEAN, FINAL_TEMPORAL_FEATURE_STD_UNBIASED_FALSE)
RESIDUAL_DIM = 128
RESIDUAL_HEAD = LINEAR_128_TO_2
NEW_TRAINABLE_PARAMETERS = 258
RESIDUAL_INITIALIZATION = EXACT_ZERO
EPOCH0 = EXACT_B0
EPOCH0_CHECKPOINT_ALLOWED = YES

MLP = NO
GATE = NO
ALPHA = NO
C0 = NO
KL = NO
DISTILLATION = NO
TEST_TIME_ADAPTATION = NO

INFERENCE_MODEL_COUNT = 1
INFERENCE_FORWARD_PASSES = 1
NEW_SEALED_TEST_ACCESSED = NO
INTERNAL_HELDOUT_STATUS = DEVELOPMENT_MODEL_SELECTION_DATA

The user explicitly selected the recovered historical 64-channel LiteBN checkpoints on the Windows server. Consequently, exact replayed historical values below replace the prompt's different same-Linux numerical reference.

- B0 outer BA: {outer_summary_row['B0_BA']:.12f}
- StatsResidual outer BA: {outer_summary_row['StatsResidual_BA']:.12f}
- StatsResidual-B0 outer: {outer_summary_row['delta_BA_pp']:+.6f} pp
- Positive/negative/tied folds: {fold_signs['positive']}/{fold_signs['negative']}/{fold_signs['tied']}
- Worst fold delta: {outer_summary_row['worst_fold_delta_pp']:+.6f} pp
- Positive/negative/tied outer subjects: {outer_signs['positive']}/{outer_signs['negative']}/{outer_signs['tied']}
- Outer subject bootstrap 95% CI: [{outer_bootstrap['ci95_low_pp']:+.6f}, {outer_bootstrap['ci95_high_pp']:+.6f}] pp
- B0 internal-heldout BA: {heldout_summary_row['B0_BA']:.12f}
- StatsResidual internal-heldout BA: {heldout_summary_row['StatsResidual_BA']:.12f}
- StatsResidual-B0 heldout: {heldout_summary_row['delta_BA_pp']:+.6f} pp
- Positive/negative/tied heldout subjects: {heldout_signs['positive']}/{heldout_signs['negative']}/{heldout_signs['tied']}
- Heldout subject bootstrap 95% CI: [{heldout_bootstrap['ci95_low_pp']:+.6f}, {heldout_bootstrap['ci95_high_pp']:+.6f}] pp
- Selected epochs: {selected_epochs}
- Folds selecting epoch0: {selected_epochs.count(0)}
- Mean selected residual-head norm: {decision['mean_selected_residual_head_norm']:.9f}
- Mean outer changed-prediction percentage: {outer_summary_row['mean_changed_prediction_percentage']:.6f}%

{terminal}
"""
        (self.out / "FINAL_STATSRES_REPORT.md").write_text(report, encoding="utf-8")
        print(terminal, flush=True)
        return terminal

    def run_all(self) -> str:
        fold_results = [self.run_fold(fold_id) for fold_id in range(5)]
        # Outer is materialized only after every fold-specific checkpoint has been selected and frozen.
        outer_rows = self.evaluate_outer()
        # Internal heldout is opened only after outer results are fully materialized.
        heldout_rows = self.evaluate_heldout()
        return self.finalize(fold_results, outer_rows, heldout_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--recovered", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--historical-runtime", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    Experiment(parse_args()).run_all()
