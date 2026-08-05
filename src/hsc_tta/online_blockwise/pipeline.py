from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from hsc_tta.models.token_heads import make_token_head
from hsc_tta.online_blockwise.core import (
    ALPHA,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    DELTA,
    GRID,
    PERMUTATION_REPETITIONS,
    PERMUTATION_SEED,
    BlockProtocol,
    ScientificStop,
    TechnicalBlock,
    block_protocol_audit,
    bootstrap_mean_ci,
    build_canonical_blocks,
    classification_metrics,
    fold_roles,
    higher_quantile,
    inclusion_indices,
    sha256_file,
    sha256_text,
    smallest_correction,
    smallest_index_for_risk,
    subject_conformal_correction,
    tps_sanity,
    tps_sets,
)


VERDICTS = {
    "STOPPED_BACKBONE_NO_GO",
    "STOPPED_BLOCK_PROTOCOL_NO_GO",
    "STOPPED_NO_DYNAMIC_HEADROOM",
    "STOPPED_BASIC_ONLINE_NO_GO",
    "STOPPED_NO_METHOD_SPACE",
    "STOPPED_TECHNICAL_BLOCK",
    "GO_TO_METHOD_DEVELOPMENT",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value.rstrip() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class V6Pipeline:
    def __init__(self, repo_root: Path, resume: bool, max_workers: int = 1):
        self.repo = repo_root.resolve()
        self.project = self.repo.parent
        self.output = self.repo / "outputs/online_blockwise_v6"
        self.delivery = self.repo / "delivery/online_blockwise_v6"
        self.audit_dir = self.output / "audit"
        self.result_dir = self.output / "results"
        self.sequence_dir = self.output / "sequence_cache"
        self.block_dir = self.output / "block_manifest"
        self.provenance = self.output / "provenance"
        self.state_path = self.output / "RUN_STATE.json"
        self.resume = resume
        self.max_workers = max_workers
        self.start = time.time()
        self.cohorts: pd.DataFrame | None = None
        self.source_models: pd.DataFrame | None = None
        self.block_manifest: pd.DataFrame | None = None
        self.sample_map: pd.DataFrame | None = None
        self.input_paths: dict[str, Path] = {}
        self.config_hash = sha256_text(json.dumps(self.freeze_payload(), sort_keys=True))
        self.state = self._load_or_initialize_state()

    def freeze_payload(self) -> dict:
        thresholds = [float(value) for value in GRID] + [1.0]
        return {
            "schema_version": "online-blockwise-v6-stage0-v2",
            "alpha": ALPHA,
            "delta": DELTA,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "permutation_repetitions": PERMUTATION_REPETITIONS,
            "permutation_seed": PERMUTATION_SEED,
            "tps_thresholds": thresholds,
            "tps_grid_hash": sha256_text(json.dumps(thresholds)),
            "K": len(thresholds) - 1,
            "hmc": {"block_epochs": 60, "tail_min_epochs": 30, "non_overlapping": True,
                    "cross_recording_blocks": False},
            "eegmmidb": {"one_original_task_run_per_block": True,
                         "minimum_valid_predictions": 8, "merge_runs": False,
                         "preserve_run_order": True, "exclude_non_task_runs": True},
            "minimum_valid_blocks_per_subject": 4,
            "datasets": ["hmc", "eegmmidb"],
            "source_seeds": list(range(5)),
            "outer_folds": list(range(5)),
            "backbone": "frozen CBraMod",
            "protected": {"formal_calibration_opened": False, "internal_final_opened": False,
                          "cap_opened": False, "full_method_entered": False},
        }

    def _load_or_initialize_state(self) -> dict:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state.get("config_hash") != self.config_hash:
                raise TechnicalBlock("V6 config hash mismatch on resume", [str(self.state_path)])
            return state
        commit = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        state = {
            "state": "INITIALIZED", "verdict": None, "terminal": False,
            "previous_state": None, "timestamp": utcnow(), "code_commit_at_run": commit,
            "config_hash": self.config_hash, "cohort_hash": None, "input_hashes": {},
            "prediction_cache_hash": None, "block_manifest_hash": None,
            "completed_jobs": [], "failed_jobs": [],
            "formal_calibration_opened": False, "internal_final_opened": False,
            "cap_opened": False, "full_method_entered": False,
        }
        atomic_json(self.state_path, state)
        return state

    def transition(self, new_state: str, **updates) -> None:
        previous = self.state["state"]
        self.state.update(updates)
        self.state.update({"previous_state": previous, "state": new_state, "timestamp": utcnow()})
        atomic_json(self.state_path, self.state)

    def stop(self, verdict: str, reason: str, evidence: list[str]) -> None:
        if verdict not in VERDICTS or verdict in {"STOPPED_TECHNICAL_BLOCK", "GO_TO_METHOD_DEVELOPMENT"}:
            raise ValueError(f"invalid scientific stop verdict: {verdict}")
        payload = {"verdict": verdict, "reason": reason, "evidence_files": evidence, "timestamp": utcnow()}
        atomic_json(self.output / "SCIENTIFIC_STOP.json", payload)
        self.state.update({"verdict": verdict, "terminal": True})
        atomic_json(self.state_path, self.state)
        raise ScientificStop(verdict, reason, evidence)

    def verify_predecessor(self) -> None:
        decision_path = self.repo / "delivery/budgeted_risk_v51_mini/MINI_DECISION.json"
        state_path = self.repo / "outputs/budgeted_risk_v51_mini/RUN_STATE.json"
        tests_path = self.repo / "outputs/budgeted_risk_v51_mini/FINAL_TESTS.txt"
        decision = json.loads(decision_path.read_text())
        prior = json.loads(state_path.read_text())
        checks = {
            "verdict": decision.get("verdict") == "MINI_STOP_FEWSHOT_FUTURE_CRITICAL_INDEX",
            "formal_calibration_closed": not decision.get("formal_calibration_opened", True),
            "internal_final_closed": not decision.get("internal_final_opened", True),
            "cap_closed": not decision.get("cap_opened", True),
            "full_method_closed": not decision.get("full_method_entered", True),
            "prior_terminal": prior.get("state") == "STOPPED",
            "tests_present": tests_path.exists(),
        }
        if not all(checks.values()):
            raise TechnicalBlock("V5.1-Mini predecessor verification failed", [str(decision_path), str(state_path)])
        write_text(self.delivery / "PREDECESSOR_LINE_CLOSURE.md",
                   "# V5.1-Mini predecessor closure\n\n" +
                   "All required predecessor checks passed. The prior few-shot future-risk route remains stopped; "
                   "V6 is a distinct online blockwise screening route.\n\n" +
                   "```json\n" + json.dumps(checks, indent=2) + "\n```")
        self.transition("PREDECESSOR_VERIFIED")

    def discover_inputs(self) -> None:
        candidates = {
            "master_cohorts": [self.repo / "outputs/contextual_risk/cohorts/MASTER_SUBJECT_COHORTS.parquet",
                               self.repo / "outputs/budgeted_risk/cohorts/MASTER_SUBJECT_COHORTS.parquet"],
            "source_cache_manifest": [self.repo / "outputs/budgeted_risk/source_cache/STAGE0_CACHE_MANIFEST.parquet"],
            "source_model_manifest": [self.repo / "outputs/budgeted_risk/source_models/STAGE0_SOURCE_MODEL_MANIFEST.parquet"],
            "episode_manifest": [self.repo / "outputs/budgeted_risk/episodes/EPISODE_MANIFEST.parquet"],
            "embedding_root": [self.project / "data/embeddings_tokens_v2"],
            "processed_root": [self.project / "data/processed"],
        }
        resolved = []
        input_hashes = {}
        for logical, options in candidates.items():
            path = next((option.resolve() for option in options if option.exists()), None)
            if path is None:
                resolved.append({"logical_input": logical, "resolved_path": None, "satisfied": False})
                continue
            self.input_paths[logical] = path
            if path.is_file():
                frame = pd.read_parquet(path)
                digest = sha256_file(path)
                resolved.append({"logical_input": logical, "resolved_path": str(path),
                                 "schema": list(frame.columns), "row_count": len(frame),
                                 "sha256": digest, "satisfied": True})
                input_hashes[str(path)] = digest
            else:
                count = sum(1 for item in path.rglob("*") if item.is_file())
                resolved.append({"logical_input": logical, "resolved_path": str(path),
                                 "schema": "directory", "row_count": count,
                                 "sha256": None, "satisfied": True})
        atomic_json(self.audit_dir / "RESOLVED_INPUT_PATHS.json", resolved)
        if not all(row.get("satisfied") for row in resolved):
            raise TechnicalBlock("Required V6 inputs could not be resolved", [str(self.audit_dir / "RESOLVED_INPUT_PATHS.json")])

        self.cohorts = pd.read_parquet(self.input_paths["master_cohorts"])
        self.source_models = pd.read_parquet(self.input_paths["source_model_manifest"])
        cohort_hash = sha256_file(self.input_paths["master_cohorts"])
        method = self.cohorts[self.cohorts.master_cohort == "method_development"]
        protected_series = self.cohorts[self.cohorts.master_cohort != "method_development"].groupby(
            ["dataset", "master_cohort"]).subject_id.nunique()
        cohort_audit = {
            "method_development_counts": method.groupby("dataset").subject_id.nunique().to_dict(),
            "protected_counts": {f"{dataset}:{cohort}": int(count)
                                 for (dataset, cohort), count in protected_series.items()},
            "screening_folds": sorted(method.screening_fold.unique().tolist()),
            "duplicates": int(self.cohorts.duplicated(["dataset", "subject_id"]).sum()),
        }
        atomic_json(self.audit_dir / "COHORT_AUDIT.json", cohort_audit)

        source_rows = []
        for row in self.source_models.itertuples(index=False):
            path = Path(row.model_path)
            actual = sha256_file(path) if path.exists() else None
            source_rows.append({"dataset": row.dataset, "fold": int(row.fold), "seed": int(row.seed),
                                "model_path": str(path), "expected_hash": row.checkpoint_sha256,
                                "actual_hash": actual, "hash_match": actual == row.checkpoint_sha256,
                                "backbone_frozen": bool(row.backbone_frozen),
                                "formal_overlap": int(row.formal_overlap),
                                "internal_final_overlap": int(row.internal_final_overlap),
                                "cap_overlap": int(row.cap_overlap),
                                "evaluation_overlap": int(row.evaluation_overlap)})
        source_audit = pd.DataFrame(source_rows)
        atomic_json(self.audit_dir / "SOURCE_MODEL_AUDIT.json", source_audit.to_dict(orient="records"))
        if not source_audit.hash_match.all() or not source_audit.backbone_frozen.all():
            raise TechnicalBlock("Frozen source model audit failed", [str(self.audit_dir / "SOURCE_MODEL_AUDIT.json")])

        time_rows = []
        for row in method.itertuples(index=False):
            suffix = row.subject_id.split(":", 1)[1]
            path = self.input_paths["processed_root"] / row.dataset / f"{row.dataset}_{suffix}.h5"
            embedding = self.input_paths["embedding_root"] / row.dataset / f"{suffix}.h5"
            if not path.exists() or not embedding.exists():
                raise TechnicalBlock(f"Missing processed/embedding file for {row.subject_id}", [str(path), str(embedding)])
            with h5py.File(path, "r") as handle, h5py.File(embedding, "r") as emb:
                labels_equal = np.array_equal(handle["label"][:], emb["labels"][:])
                windows_equal = np.array_equal(np.arange(len(handle["label"])), emb["window_indices"][:])
                time_rows.append({"dataset": row.dataset, "subject_id": row.subject_id,
                                  "n_samples": int(len(handle["label"])),
                                  "recording_available": "recording_id" in handle,
                                  "run_available": "run_id" in handle,
                                  "time_available": "window_start" in handle,
                                  "labels_aligned": labels_equal, "window_indices_aligned": windows_equal})
        time_audit = pd.DataFrame(time_rows)
        atomic_json(self.audit_dir / "TIME_ORDER_AUDIT.json", time_audit.to_dict(orient="records"))
        if not time_audit[["recording_available", "run_available", "time_available",
                           "labels_aligned", "window_indices_aligned"]].all().all():
            raise TechnicalBlock("Time-order audit failed", [str(self.audit_dir / "TIME_ORDER_AUDIT.json")])
        atomic_json(self.audit_dir / "INPUT_HASHES.json", input_hashes)
        write_text(self.delivery / "INPUT_AND_LEAKAGE_AUDIT.md",
                   "# Input and leakage audit\n\n"
                   f"Resolved all required inputs. Method-development subjects: {cohort_audit['method_development_counts']}. "
                   "Only these subjects are eligible for sequence cache generation. All 50 source heads are frozen and "
                   "their checkpoint hashes match. Processed labels align exactly with frozen token embeddings. "
                   "Formal/internal/CAP cohorts are excluded by the master cohort filter.")
        self.state.update({"cohort_hash": cohort_hash, "input_hashes": input_hashes})
        self.transition("INPUT_AUDIT_COMPLETE")

    def freeze_protocol(self) -> None:
        payload = self.freeze_payload()
        atomic_json(self.delivery / "V6_STAGE0_FREEZE.json", payload)
        write_text(self.delivery / "V6_STAGE0_PROTOCOL.md",
                   "# V6 Stage-0 frozen protocol\n\n"
                   "The online blockwise protocol was frozen before reading Oracle or online-policy results.\n\n"
                   "```json\n" + json.dumps(payload, indent=2) + "\n```")
        self.transition("PROTOCOL_FROZEN")

    def _model_row(self, dataset: str, fold: int, seed: int):
        selected = self.source_models[(self.source_models.dataset == dataset) &
                                      (self.source_models.fold == fold) &
                                      (self.source_models.seed == seed)]
        if len(selected) != 1:
            raise TechnicalBlock(f"Expected one source model for {dataset}/{fold}/{seed}", [])
        return selected.iloc[0]

    def build_sequence_cache(self) -> None:
        manifest_path = self.sequence_dir / "PREDICTION_CACHE_MANIFEST.parquet"
        expected_jobs = {f"{d}:{f}:{s}" for d in ("hmc", "eegmmidb") for f in range(5) for s in range(5)}
        completed = set(self.state.get("completed_jobs", []))
        if expected_jobs.issubset(completed) and manifest_path.exists():
            return
        method = self.cohorts[self.cohorts.master_cohort == "method_development"].copy()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        manifest_rows = []
        if manifest_path.exists():
            manifest_rows = pd.read_parquet(manifest_path).to_dict(orient="records")
        for dataset in ("hmc", "eegmmidb"):
            subjects = method[method.dataset == dataset].sort_values("subject_id")
            n_classes = 5 if dataset == "hmc" else 4
            for fold in range(5):
                for seed in range(5):
                    job = f"{dataset}:{fold}:{seed}"
                    if job in completed:
                        continue
                    model_row = self._model_row(dataset, fold, seed)
                    checkpoint = torch.load(model_row.model_path, map_location="cpu", weights_only=False)
                    model = make_token_head(checkpoint["architecture"], n_classes)
                    model.load_state_dict(checkpoint["state_dict"], strict=True)
                    model.eval().to(device)
                    for subject_row in subjects.itertuples(index=False):
                        suffix = subject_row.subject_id.split(":", 1)[1]
                        embedding_path = self.input_paths["embedding_root"] / dataset / f"{suffix}.h5"
                        processed_path = self.input_paths["processed_root"] / dataset / f"{dataset}_{suffix}.h5"
                        with h5py.File(embedding_path, "r") as emb, h5py.File(processed_path, "r") as proc:
                            n = len(emb["labels"])
                            probabilities = []
                            batch_size = 256 if dataset == "hmc" else 128
                            with torch.inference_mode():
                                for start in range(0, n, batch_size):
                                    tokens = torch.from_numpy(emb["token_embeddings"][start:start + batch_size].astype("float32"))
                                    logits = model(tokens.to(device)).float().cpu()
                                    probabilities.append(torch.softmax(logits, dim=1).numpy())
                            p = np.concatenate(probabilities, axis=0)
                            labels = emb["labels"][:].astype(int)
                            indices = emb["window_indices"][:].astype(int)
                            recording = proc["recording_id"][:].astype(str)
                            run_id = proc["run_id"][:].astype(int)
                            starts = proc["window_start"][:].astype(float)
                            ends = proc["window_end"][:].astype(float)
                            backbone_hash = str(emb.attrs["checkpoint_sha256"])
                            raw_hash = str(emb.attrs["raw_source_hash"])
                        frame = pd.DataFrame({
                            "dataset": dataset, "subject_id": subject_row.subject_id,
                            "recording_id": recording,
                            "sample_id": [f"{subject_row.subject_id}:{int(i):06d}" for i in indices],
                            "chronological_index": indices,
                            "timestamp_or_sequence_position": starts,
                            "window_end": ends,
                            "run_id": run_id,
                            "epoch_index": indices,
                            "label": labels,
                            "outer_fold": fold, "source_seed": seed,
                            "source_model_hash": model_row.checkpoint_sha256,
                            "backbone_hash": backbone_hash,
                            "input_sample_hash": [sha256_text(f"{raw_hash}:{int(i)}") for i in indices],
                        })
                        for class_index in range(n_classes):
                            frame[f"probability_{class_index}"] = p[:, class_index]
                        path = self.sequence_dir / dataset / f"fold_{fold}" / f"seed_{seed}" / f"{subject_row.subject_id.replace(':', '_')}.parquet"
                        atomic_parquet(frame, path)
                        manifest_rows.append({"dataset": dataset, "outer_fold": fold, "source_seed": seed,
                                              "subject_id": subject_row.subject_id, "screening_fold": int(subject_row.screening_fold),
                                              "cache_path": str(path), "row_count": len(frame),
                                              "cache_sha256": sha256_file(path),
                                              "source_model_hash": model_row.checkpoint_sha256,
                                              "backbone_hash": backbone_hash})
                    completed.add(job)
                    self.state["completed_jobs"] = sorted(completed)
                    atomic_parquet(pd.DataFrame(manifest_rows).drop_duplicates(
                        ["dataset", "outer_fold", "source_seed", "subject_id"], keep="last"), manifest_path)
                    atomic_json(self.state_path, self.state)
                    if device.type == "cuda":
                        model.cpu()
                        torch.cuda.empty_cache()
        manifest = pd.read_parquet(manifest_path).sort_values(
            ["dataset", "outer_fold", "source_seed", "subject_id"])
        cache_hash = sha256_text("\n".join(manifest.cache_sha256.astype(str)))
        self.transition("SEQUENCE_CACHE_COMPLETE", prediction_cache_hash=cache_hash,
                        sequence_cache_device=str(device), sequence_cache_rebuilt=True)

    def _cache(self, dataset: str, fold: int, seed: int, subject: str) -> pd.DataFrame:
        path = self.sequence_dir / dataset / f"fold_{fold}" / f"seed_{seed}" / f"{subject.replace(':', '_')}.parquet"
        return pd.read_parquet(path)

    def audit_backbone(self) -> None:
        subject_rows = []
        seed_rows = []
        sanity_rows = []
        method = self.cohorts[self.cohorts.master_cohort == "method_development"]
        overlap_max = {"hmc": 0.0, "eegmmidb": 0.0}
        for dataset in ("hmc", "eegmmidb"):
            n_classes = 5 if dataset == "hmc" else 4
            subjects = method[method.dataset == dataset]
            for seed in range(5):
                pooled_p, pooled_y = [], []
                for row in subjects.itertuples(index=False):
                    fold = int(row.screening_fold)
                    frame = self._cache(dataset, fold, seed, row.subject_id)
                    probability_columns = [f"probability_{i}" for i in range(n_classes)]
                    p = frame[probability_columns].to_numpy()
                    y = frame.label.to_numpy(dtype=int)
                    metrics = classification_metrics(p, y, n_classes)
                    subject_rows.append({"dataset": dataset, "source_seed": seed,
                                         "subject_id": row.subject_id, "screening_fold": fold, **metrics})
                    pooled_p.append(p); pooled_y.append(y)

                    old_path = self.repo / "outputs/budgeted_risk/source_cache/stage0" / dataset / f"fold_{fold}" / f"seed_{seed}" / f"{row.subject_id.replace(':', '_')}.npz"
                    if old_path.exists():
                        with np.load(old_path, allow_pickle=False) as old:
                            old_indices = np.concatenate([old["context_sample_indices"], old["future_sample_indices"]]).astype(int)
                            old_p = np.concatenate([old["context_probabilities"], old["future_probabilities"]])
                        overlap_max[dataset] = max(overlap_max[dataset], float(np.max(np.abs(p[old_indices] - old_p))))
                pooled_p = np.concatenate(pooled_p); pooled_y = np.concatenate(pooled_y)
                metrics = classification_metrics(pooled_p, pooled_y, n_classes)
                seed_rows.append({"dataset": dataset, "source_seed": seed, **metrics})
                sanity_rows.append({"dataset": dataset, "source_seed": seed, **tps_sanity(pooled_p, pooled_y)})

        subject_metrics = pd.DataFrame(subject_rows)
        seed_metrics = pd.DataFrame(seed_rows)
        sanity = pd.DataFrame(sanity_rows)
        atomic_parquet(subject_metrics, self.result_dir / "BACKBONE_METRICS_BY_SUBJECT.parquet")
        atomic_csv(seed_metrics, self.result_dir / "BACKBONE_METRICS_BY_SEED.csv")
        atomic_csv(sanity, self.result_dir / "TPS_SANITY.csv")

        gate_rows = []
        for dataset in ("hmc", "eegmmidb"):
            ds_seed = seed_metrics[seed_metrics.dataset == dataset]
            ds_subject = subject_metrics[subject_metrics.dataset == dataset]
            ds_sanity = sanity[sanity.dataset == dataset]
            n_classes = 5 if dataset == "hmc" else 4
            all_classes = all(ds_seed[f"support_{i}"].sum() > 0 for i in range(n_classes))
            frequency = all(ds_seed[f"predicted_frequency_{i}"].mean() >= .001 for i in range(n_classes))
            sanity_columns = ["finite", "nonnegative", "row_sum", "labels_legal", "nested", "size_monotone",
                              "coverage_monotone", "inclusion_index_legal", "full_index", "nonfull_informative"]
            row = {
                "dataset": dataset,
                "probability_tps_sanity": bool(ds_sanity[sanity_columns].all().all() and
                                                (ds_sanity.distinct_mean_sizes >= 3).all() and
                                                (ds_sanity.distinct_coverages >= 3).all()),
                "all_true_classes": bool(all_classes),
                "predicted_frequency": bool(frequency),
                "dataset_balanced_accuracy": float(ds_seed.balanced_accuracy.mean()),
                "dataset_balanced_accuracy_pass": bool(ds_seed.balanced_accuracy.mean() >= (.30 if dataset == "hmc" else .35)),
                "median_subject_balanced_accuracy": float(ds_subject.groupby("subject_id").balanced_accuracy.mean().median()),
                "median_subject_balanced_accuracy_pass": bool(ds_subject.groupby("subject_id").balanced_accuracy.mean().median() >= (.25 if dataset == "hmc" else .30)),
                "seed_balanced_accuracy_std": float(ds_seed.balanced_accuracy.std(ddof=0)),
                "seed_stability_pass": bool(ds_seed.balanced_accuracy.std(ddof=0) <= .05),
                "nonconstant_subject_rate": float(1 - ds_subject.groupby("subject_id").constant_prediction.all().mean()),
                "nonconstant_subject_pass": bool(1 - ds_subject.groupby("subject_id").constant_prediction.all().mean() >= .90),
                "v5_overlap_max_abs_diff": overlap_max[dataset],
                "v5_overlap_pass": bool(overlap_max[dataset] <= 1e-6),
            }
            row["pass"] = all(value for key, value in row.items() if key.endswith("_pass") or key in {
                "probability_tps_sanity", "all_true_classes", "predicted_frequency"})
            gate_rows.append(row)
        gate = pd.DataFrame(gate_rows)
        atomic_csv(gate, self.result_dir / "BACKBONE_GATE.csv")
        write_text(self.delivery / "BACKBONE_OUTPUT_AUDIT.md",
                   "# Frozen CBraMod output audit\n\n" + gate.to_markdown(index=False) +
                   "\n\nMetrics use each subject exactly once at its screening-fold OOF head, separately for five source seeds. "
                   "The full sequence cache was rebuilt from frozen token embeddings with frozen heads; no backward or optimizer was used.")
        self.transition("BACKBONE_AUDIT_COMPLETE")
        if not gate["pass"].all():
            self.stop("STOPPED_BACKBONE_NO_GO", "At least one dataset failed the preregistered frozen-backbone gate.",
                      [str(self.result_dir / "BACKBONE_GATE.csv"), str(self.delivery / "BACKBONE_OUTPUT_AUDIT.md")])

    def build_block_manifest(self) -> None:
        method = self.cohorts[self.cohorts.master_cohort == "method_development"]
        rows = []
        for row in method.itertuples(index=False):
            suffix = row.subject_id.split(":", 1)[1]
            path = self.input_paths["processed_root"] / row.dataset / f"{row.dataset}_{suffix}.h5"
            with h5py.File(path, "r") as handle:
                n = len(handle["label"])
                rows.append(pd.DataFrame({
                    "dataset": row.dataset, "subject_id": row.subject_id,
                    "screening_fold": int(row.screening_fold),
                    "recording_id": handle["recording_id"][:].astype(str),
                    "run_id": handle["run_id"][:].astype(int),
                    "chronological_index": np.arange(n, dtype=int),
                    "sample_id": [f"{row.subject_id}:{i:06d}" for i in range(n)],
                    "window_start": handle["window_start"][:], "window_end": handle["window_end"][:],
                }))
        metadata = pd.concat(rows, ignore_index=True)
        blocks, sample_map = build_canonical_blocks(metadata, BlockProtocol())
        atomic_parquet(blocks, self.block_dir / "BLOCK_MANIFEST.parquet")
        atomic_parquet(sample_map, self.block_dir / "BLOCK_SAMPLE_MAP.parquet")
        gate = block_protocol_audit(blocks, self.cohorts)
        atomic_csv(gate, self.result_dir / "BLOCK_PROTOCOL_GATE.csv")
        digest = sha256_file(self.block_dir / "BLOCK_MANIFEST.parquet")
        self.block_manifest, self.sample_map = blocks, sample_map
        write_text(self.delivery / "BLOCK_PROTOCOL_AUDIT.md",
                   "# Canonical block protocol audit\n\n" + gate.to_markdown(index=False) +
                   "\n\nBlocks are canonical physical structures and are not duplicated by source seed. HMC blocks are contiguous, "
                   "non-overlapping, and never cross recordings. EEGMMIDB blocks preserve one original task run each.")
        self.transition("BLOCK_MANIFEST_COMPLETE", block_manifest_hash=digest)
        if not gate["pass"].all():
            self.stop("STOPPED_BLOCK_PROTOCOL_NO_GO", "At least one dataset failed the preregistered block protocol gate.",
                      [str(self.result_dir / "BLOCK_PROTOCOL_GATE.csv"), str(self.delivery / "BLOCK_PROTOCOL_AUDIT.md")])

    def _valid_subjects(self, dataset: str) -> set[str]:
        blocks = self.block_manifest[(self.block_manifest.dataset == dataset) & self.block_manifest.retained]
        counts = blocks.groupby("subject_id").size()
        return set(counts[counts >= 4].index)

    def _subject_arrays(self, dataset: str, fold: int, seed: int, subject: str):
        frame = self._cache(dataset, fold, seed, subject)
        mapping = self.sample_map[(self.sample_map.dataset == dataset) & (self.sample_map.subject_id == subject)]
        merged = mapping.merge(frame, on=["dataset", "subject_id", "recording_id", "sample_id"], validate="one_to_one")
        n_classes = 5 if dataset == "hmc" else 4
        p = merged[[f"probability_{i}" for i in range(n_classes)]].to_numpy()
        y = merged.label.to_numpy(dtype=int)
        sets = tps_sets(p)
        kappa = inclusion_indices(sets, y)
        sizes = sets.sum(axis=2)
        blocks = []
        for block_id, block in merged.groupby("block_id", sort=False):
            index = block.index.to_numpy()
            blocks.append((block_id, kappa[index], sizes[index]))
        return kappa, sizes, blocks

    def oracle_screening(self) -> None:
        K = int(self.freeze_payload()["K"])
        global_rows, subject_rows, block_rows, permuted_rows = [], [], [], []
        rng = np.random.default_rng(PERMUTATION_SEED)
        for dataset in ("hmc", "eegmmidb"):
            valid = self._valid_subjects(dataset)
            cohort = self.cohorts[(self.cohorts.dataset == dataset) &
                                  (self.cohorts.master_cohort == "method_development") &
                                  (self.cohorts.subject_id.isin(valid))]
            for fold in range(5):
                roles = fold_roles(fold)
                dev = cohort[cohort.screening_fold.isin(roles["development"])]
                cal = cohort[cohort.screening_fold.isin(roles["calibration"])]
                evaluation = cohort[cohort.screening_fold == fold]
                for seed in range(5):
                    arrays = {row.subject_id: self._subject_arrays(dataset, fold, seed, row.subject_id)
                              for row in cohort.itertuples(index=False)}
                    raw_global = K
                    for candidate in range(K + 1):
                        risks = [float(np.mean(arrays[s][0] > candidate)) for s in dev.subject_id]
                        if np.mean(risks) <= ALPHA:
                            raw_global = candidate; break
                    qs = [smallest_correction(arrays[s][0], raw_global, ALPHA, K) for s in cal.subject_id]
                    q_global, m, rank, sentinel = subject_conformal_correction(qs, DELTA, K)
                    certified_global = min(K, raw_global + q_global)
                    for subject in evaluation.subject_id:
                        kappa, sizes, blocks = arrays[subject]
                        global_size = float(sizes[:, certified_global].mean())
                        global_risk = float(np.mean(kappa > certified_global))
                        subject_index = smallest_index_for_risk(kappa, ALPHA, K)
                        subject_size = float(sizes[:, subject_index].mean())
                        subject_risk = float(np.mean(kappa > subject_index))
                        block_indices, block_sizes, block_risks = [], [], []
                        for block_id, block_kappa, block_size_curve in blocks:
                            block_index = smallest_index_for_risk(block_kappa, ALPHA, K)
                            size = float(block_size_curve[:, block_index].mean())
                            risk = float(np.mean(block_kappa > block_index))
                            block_indices.append(block_index); block_sizes.append(size); block_risks.append(risk)
                            block_rows.append({"dataset": dataset, "outer_fold": fold, "source_seed": seed,
                                               "subject_id": subject, "block_id": block_id,
                                               "block_oracle_index": block_index, "average_set_size": size,
                                               "miscoverage": risk, "n_samples": len(block_kappa)})
                        block_size = float(np.average(block_sizes, weights=[len(b[1]) for b in blocks]))
                        block_risk = float(np.average(block_risks, weights=[len(b[1]) for b in blocks]))
                        global_rows.append({"dataset": dataset, "outer_fold": fold, "source_seed": seed,
                                            "subject_id": subject, "raw_global_index": raw_global,
                                            "q_global": q_global, "calibration_m": m, "conformal_rank_k": rank,
                                            "finite_sample_sentinel": sentinel,
                                            "certified_global_index": certified_global,
                                            "average_set_size": global_size, "miscoverage": global_risk})
                        subject_rows.append({"dataset": dataset, "outer_fold": fold, "source_seed": seed,
                                             "subject_id": subject, "subject_oracle_index": subject_index,
                                             "average_set_size": subject_size, "miscoverage": subject_risk})

                        concatenated_kappa = np.concatenate([b[1] for b in blocks])
                        concatenated_sizes = np.concatenate([b[2] for b in blocks], axis=0)
                        block_lengths = [len(b[1]) for b in blocks]
                        for repetition in range(PERMUTATION_REPETITIONS):
                            permutation = rng.permutation(len(concatenated_kappa))
                            cursor = 0; total_size = 0.0
                            for length in block_lengths:
                                chosen = permutation[cursor:cursor + length]; cursor += length
                                index = smallest_index_for_risk(concatenated_kappa[chosen], ALPHA, K)
                                total_size += float(concatenated_sizes[chosen, index].sum())
                            permuted_size = total_size / len(concatenated_kappa)
                            permuted_rows.append({"dataset": dataset, "outer_fold": fold, "source_seed": seed,
                                                  "subject_id": subject, "repetition": repetition,
                                                  "average_set_size": permuted_size,
                                                  "gain_vs_global": (global_size - permuted_size) / global_size})

        global_frame = pd.DataFrame(global_rows)
        subject_frame = pd.DataFrame(subject_rows)
        block_frame = pd.DataFrame(block_rows)
        permuted_frame = pd.DataFrame(permuted_rows)
        atomic_parquet(global_frame, self.result_dir / "STATIC_GLOBAL_RESULTS.parquet")
        atomic_parquet(subject_frame, self.result_dir / "SUBJECT_ORACLE_RESULTS.parquet")
        atomic_parquet(block_frame, self.result_dir / "BLOCK_ORACLE_RESULTS.parquet")
        atomic_parquet(permuted_frame, self.result_dir / "PERMUTED_BLOCK_ORACLE_RESULTS.parquet")

        block_subject = block_frame.groupby(["dataset", "source_seed", "subject_id"]).apply(
            lambda f: pd.Series({"block_size": np.average(f.average_set_size, weights=f.n_samples),
                                 "distinct_j": f.block_oracle_index.nunique()}), include_groups=False).reset_index()
        merged = global_frame.merge(subject_frame, on=["dataset", "outer_fold", "source_seed", "subject_id"],
                                    suffixes=("_global", "_subject")).merge(
            block_subject, on=["dataset", "source_seed", "subject_id"])
        perm = permuted_frame.groupby(["dataset", "source_seed", "subject_id"]).gain_vs_global.median().rename(
            "permuted_gain").reset_index()
        merged = merged.merge(perm, on=["dataset", "source_seed", "subject_id"])
        merged["subject_gain"] = (merged.average_set_size_global - merged.average_set_size_subject) / merged.average_set_size_global
        merged["block_gain"] = (merged.average_set_size_global - merged.block_size) / merged.average_set_size_global
        merged["block_vs_subject_gain"] = (merged.average_set_size_subject - merged.block_size) / merged.average_set_size_global
        merged["contiguous_minus_permuted"] = merged.block_gain - merged.permuted_gain
        atomic_parquet(merged, self.result_dir / "RESULTS_BY_SEED.parquet")

        across_seed = merged.groupby(["dataset", "subject_id"], as_index=False).agg(
            global_size=("average_set_size_global", "mean"), subject_size=("average_set_size_subject", "mean"),
            block_size=("block_size", "mean"), subject_gain=("subject_gain", "mean"),
            block_gain=("block_gain", "mean"), block_vs_subject_gain=("block_vs_subject_gain", "mean"),
            permuted_gain=("permuted_gain", "mean"), contiguous_minus_permuted=("contiguous_minus_permuted", "mean"),
            distinct_j=("distinct_j", "mean"))
        atomic_parquet(across_seed, self.result_dir / "RESULTS_BY_SUBJECT.parquet")
        summary_rows, gate_rows = [], []
        for dataset in ("hmc", "eegmmidb"):
            ds = across_seed[across_seed.dataset == dataset]
            ds_seed = merged[merged.dataset == dataset].groupby("source_seed").block_gain.mean()
            ci = bootstrap_mean_ci(ds.block_gain)
            diff_ci = bootstrap_mean_ci(ds.contiguous_minus_permuted)
            loo_positive = all(ds.drop(index=index).block_gain.mean() > 0 for index in ds.index)
            summary = {
                "dataset": dataset, "n_subjects": len(ds), "static_global_size": ds.global_size.mean(),
                "subject_oracle_size": ds.subject_size.mean(), "block_oracle_size": ds.block_size.mean(),
                "subject_oracle_gain": ds.subject_gain.mean(), "block_oracle_gain": ds.block_gain.mean(),
                "block_oracle_gain_ci_low": ci[0], "block_oracle_gain_ci_high": ci[1],
                "block_vs_subject_gain": ds.block_vs_subject_gain.mean(),
                "permuted_oracle_gain": ds.permuted_gain.mean(),
                "contiguous_minus_permuted": ds.contiguous_minus_permuted.mean(),
                "contiguous_minus_permuted_ci_low": diff_ci[0],
                "contiguous_minus_permuted_ci_high": diff_ci[1],
                "positive_gain_subject_rate": float(np.mean(ds.block_gain > 0)),
                "dynamic_subject_rate": float(np.mean(ds.distinct_j >= 2)),
                "positive_seed_count": int(np.sum(ds_seed > 0)),
                "loo_positive": bool(loo_positive),
                "global_full_set_rate": float(np.mean(global_frame[global_frame.dataset == dataset].certified_global_index == int(self.freeze_payload()["K"]))),
            }
            summary_rows.append(summary)
            gates = {"A1": summary["block_oracle_gain"] >= .10,
                     "A2": summary["block_oracle_gain_ci_low"] > 0,
                     "A3": summary["positive_gain_subject_rate"] >= .70,
                     "A4": summary["block_vs_subject_gain"] >= .03,
                     "A5": summary["contiguous_minus_permuted"] >= .02,
                     "A6": summary["contiguous_minus_permuted_ci_low"] > 0,
                     "A7": summary["dynamic_subject_rate"] >= .50,
                     "A8": summary["positive_seed_count"] >= 4,
                     "A9": summary["loo_positive"]}
            gate_rows.append({"dataset": dataset, **gates, "pass": all(gates.values())})
        summary_frame = pd.DataFrame(summary_rows)
        gate_frame = pd.DataFrame(gate_rows)
        atomic_csv(summary_frame, self.result_dir / "ORACLE_HEADROOM_SUMMARY.csv")
        atomic_csv(gate_frame, self.result_dir / "GATE_A.csv")
        write_text(self.delivery / "ORACLE_HEADROOM_REPORT.md", "# Oracle headroom\n\n" + summary_frame.to_markdown(index=False))
        write_text(self.delivery / "PERMUTATION_NULL_REPORT.md",
                   "# Permutation null\n\nThe null preserves subject membership, sample-label-probability pairing, and each true block-size vector.\n\n" +
                   summary_frame[["dataset", "permuted_oracle_gain", "contiguous_minus_permuted",
                                  "contiguous_minus_permuted_ci_low", "contiguous_minus_permuted_ci_high"]].to_markdown(index=False))
        write_text(self.delivery / "GATE_A_DECISION.md", "# Gate A decision\n\n" + gate_frame.to_markdown(index=False))
        self.transition("ORACLE_SCREENING_COMPLETE")
        if not gate_frame["pass"].all():
            self.stop("STOPPED_NO_DYNAMIC_HEADROOM", "At least one dataset failed at least one preregistered Gate A condition.",
                      [str(self.result_dir / "GATE_A.csv"), str(self.delivery / "GATE_A_DECISION.md")])
        raise TechnicalBlock("Gate A passed, but D1 baseline execution is not yet present in this build", [str(self.result_dir / "GATE_A.csv")])

    def build_delivery(self, verdict: str) -> None:
        gate_files = sorted(str(path.relative_to(self.repo)) for path in self.result_dir.glob("*GATE*.csv"))
        decision = {"verdict": verdict, "state": "STOPPED", "terminal": True,
                    "runtime_seconds": time.time() - self.start,
                    "formal_calibration_opened": False, "internal_final_opened": False,
                    "cap_opened": False, "full_method_entered": False,
                    "sleep_edf_run": False, "bcic2a_run": False, "other_backbone_run": False,
                    "sequence_cache_rebuilt": bool(self.state.get("sequence_cache_rebuilt", False)),
                    "gpu_used": self.state.get("sequence_cache_device") == "cuda",
                    "gate_files": gate_files}
        atomic_json(self.delivery / "V6_STAGE0_DECISION.json", decision)
        write_text(self.delivery / "V6_STAGE0_DECISION.md",
                   "# V6 Stage-0 decision\n\n"
                   f"Final verdict: `{verdict}`.\n\n"
                   "Formal calibration, internal final, CAP, Sleep-EDF, BCIC2A, other backbones, and full method development "
                   "were not opened. Later stages after the stopping gate were not run.")
        write_text(self.delivery / "LIMITATIONS.md",
                   "# Limitations\n\nThis is a Stage-0 screening experiment on frozen HMC and EEGMMIDB method-development cohorts. "
                   "Oracle analyses are upper-bound diagnostics, not deployable methods. No result supports access to protected cohorts.")
        write_text(self.delivery / "REPRODUCE.md",
                   "# Reproduce\n\n```bash\n/root/miniconda3/envs/hsc_gpu/bin/python scripts/online_blockwise_v6/run_all.py \\\n+  --repo-root /root/autodl-tmp/hsc_tta_eeg/repo --resume\n```")
        write_text(
            self.delivery / "REPRODUCE.md",
            "# Reproduce\n\n```bash\n/root/miniconda3/envs/hsc_gpu/bin/python "
            "scripts/online_blockwise_v6/run_all.py --repo-root "
            "/root/autodl-tmp/hsc_tta_eeg/repo --resume\n```",
        )
        files = []
        for root in (self.output, self.delivery):
            for path in root.rglob("*"):
                if path.is_file() and not path.name.endswith(".tmp"):
                    files.append({"path": str(path.relative_to(self.repo)), "bytes": path.stat().st_size,
                                  "sha256": sha256_file(path)})
        atomic_json(self.delivery / "DELIVERY_MANIFEST.json", {"created_at": utcnow(), "files": files})
        self.state.update({"verdict": verdict, "terminal": True})
        self.transition("DELIVERY_COMPLETE")
        self.transition("STOPPED", verdict=verdict, terminal=True)

    def run(self) -> str:
        if self.state.get("terminal"):
            verdict = self.state["verdict"]
            self.build_delivery(verdict)
            return verdict
        try:
            self.verify_predecessor()
            self.discover_inputs()
            self.freeze_protocol()
            self.build_sequence_cache()
            self.audit_backbone()
            self.build_block_manifest()
            self.oracle_screening()
            raise AssertionError("oracle screening must stop or continue explicitly")
        except ScientificStop as stop:
            self.build_delivery(stop.verdict)
            return stop.verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--stage")
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args(argv)
    if args.stage or args.dataset or args.seed is not None or args.fold is not None or args.audit_only or args.oracle_only or args.baseline_only:
        print("Scoped execution flags are accepted only through a validated resume state; use --resume for the complete gated run.", file=sys.stderr)
        return 2
    try:
        pipeline = V6Pipeline(args.repo_root, args.resume, args.max_workers)
        verdict = pipeline.run()
        print(verdict)
        return 0
    except TechnicalBlock as block:
        output = args.repo_root / "outputs/online_blockwise_v6"
        output.mkdir(parents=True, exist_ok=True)
        payload = {"verdict": "STOPPED_TECHNICAL_BLOCK", "reason": block.reason,
                   "evidence_files": block.evidence_files, "timestamp": utcnow()}
        atomic_json(output / "TECHNICAL_BLOCK.json", payload)
        state_path = output / "RUN_STATE.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            state.update({"state": "STOPPED", "verdict": "STOPPED_TECHNICAL_BLOCK",
                          "terminal": True, "timestamp": utcnow()})
            atomic_json(state_path, state)
        print(block.reason, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
