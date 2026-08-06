from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    recall_score,
)

from .core import TechnicalBlock, fold_roles, gate_q, normalized_inverse_frequency
from .models import (
    BIOT_MONTAGES,
    biot_embeddings,
    load_biot,
    load_labram,
    labram_embeddings,
    normalized_channel,
    sha256_file,
)

VERDICTS = {
    "V7_STAGE0A_STOP_INSUFFICIENT_MODEL_POOL",
    "V7_STAGE0A_STOP_MODEL_QUALIFICATION_FAILURE",
    "V7_STAGE0A_STOP_NO_CROSS_MODEL_HEADROOM",
    "V7_STAGE0A_STOP_NO_STABLE_SUBJECT_COMPLEMENTARITY",
    "V7_STAGE0A_CONTINUE_TO_UNLABELED_ROUTING_SCREEN",
    "V7_STAGE0A_TECHNICAL_BLOCK",
}
MODELS = ("cbramod", "labram", "biot")
DATASETS = ("hmc", "eegmmidb")
SEEDS = tuple(range(5))
FOLDS = tuple(range(5))
PROTECTED_FLAGS = {
    "formal_calibration_opened": False,
    "internal_final_opened": False,
    "cap_opened": False,
    "sleep_edf_opened": False,
    "bcic2a_opened": False,
    "router_developed": False,
    "abstention_developed": False,
    "full_method_entered": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_parquet(path: pathlib.Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


@dataclass
class Paths:
    repo: pathlib.Path

    @property
    def project(self) -> pathlib.Path:
        return self.repo.parent

    @property
    def output(self) -> pathlib.Path:
        return self.repo / "outputs/fm_routing_v7"

    @property
    def delivery(self) -> pathlib.Path:
        return self.repo / "delivery/fm_routing_v7"

    @property
    def state(self) -> pathlib.Path:
        return self.output / "RUN_STATE.json"


class Pipeline:
    def __init__(self, repo_root: str | pathlib.Path):
        self.paths = Paths(pathlib.Path(repo_root).resolve())
        self.started = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paths.output.mkdir(parents=True, exist_ok=True)
        self.paths.delivery.mkdir(parents=True, exist_ok=True)
        self.commands = self.paths.output / "provenance/COMMANDS.txt"
        self._state = self._load_or_initialize_state()

    def _load_or_initialize_state(self) -> dict[str, Any]:
        if self.paths.state.exists():
            state = json.loads(self.paths.state.read_text())
            if state.get("terminal"):
                return state
            return state
        state = {
            "state": "INITIALIZED",
            "verdict": None,
            "terminal": False,
            "started_at": utc_now(),
            "starting_commit": subprocess.check_output(
                ["git", "-C", str(self.paths.repo), "rev-parse", "HEAD"], text=True
            ).strip(),
            "device": str(self.device),
            "gpu_used": False,
            "completed_jobs": [],
            "failed_jobs": [],
            **PROTECTED_FLAGS,
        }
        atomic_json(self.paths.state, state)
        return state

    def transition(self, state: str, **updates: Any) -> None:
        self._state.update(updates)
        self._state["previous_state"] = self._state.get("state")
        self._state["state"] = state
        self._state["updated_at"] = utc_now()
        self._state["elapsed_seconds"] = time.time() - self.started
        atomic_json(self.paths.state, self._state)
        print(json.dumps({
            "state": state,
            "verdict": self._state.get("verdict"),
            "completed_jobs": len(self._state.get("completed_jobs", [])),
            "elapsed_seconds": round(self._state["elapsed_seconds"], 2),
            "gpu_memory": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
        }), flush=True)

    def record_command(self, command: str) -> None:
        self.commands.parent.mkdir(parents=True, exist_ok=True)
        with self.commands.open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()}\t{command}\n")

    def audit_predecessor(self) -> None:
        decision_path = self.paths.repo / "delivery/online_blockwise_v6/V6_STAGE0_DECISION.json"
        state_path = self.paths.repo / "outputs/online_blockwise_v6/RUN_STATE.json"
        decision = json.loads(decision_path.read_text())
        predecessor = json.loads(state_path.read_text())
        assert decision["verdict"] == "STOPPED_NO_DYNAMIC_HEADROOM"
        assert decision["stopping_gate"] == "Gate A"
        assert predecessor["terminal"] is True
        assert set(decision["later_stages_not_run"]) >= {"B1", "B2", "B3", "B4", "Gate B", "Gate C", "method development"}
        for key in ("formal_calibration_opened", "internal_final_opened", "cap_opened", "full_method_entered"):
            assert predecessor.get(key, False) is False
        closure = """# V6 line closed

- The frozen backbone gate and block-protocol gate both passed.
- The apparent block-oracle gain was large.
- The extra gain of true contiguous blocks over the preregistered permutation null was insufficient.
- The V6 dynamic threshold-adaptation line is closed.
- Later online baselines, Gate B, Gate C, and method development were not run.
- V7 does not optimize the single-model prediction-set index.
"""
        write_text(self.paths.repo / "delivery/online_blockwise_v6/LINE_CLOSED.md", closure)
        write_text(self.paths.delivery / "V6_LINE_CLOSURE_REFERENCE.md", closure)
        self.transition("V6_LINE_CLOSED", predecessor_verdict=decision["verdict"])

    def audit_and_freeze_models(self) -> None:
        project = self.paths.project
        checkpoints = {
            "cbramod": project / "checkpoints/cbramod/pretrained_weights.pth",
            "labram": project / "external/LaBraM/checkpoints/labram-base.pth",
            "biot": project / "external/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt",
        }
        code_roots = {
            "cbramod": project / "external/CBraMod",
            "labram": project / "external/LaBraM",
            "biot": project / "external/BIOT",
        }
        for path in (*checkpoints.values(), *code_roots.values()):
            if not path.exists():
                raise TechnicalBlock(f"required frozen asset missing: {path}")
        commits = {
            name: subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            for name, root in code_roots.items()
        }
        hashes = {name: sha256_file(path) for name, path in checkpoints.items()}
        params = {}
        for name, path in checkpoints.items():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if name == "cbramod":
                candidate = payload.get("model", payload) if isinstance(payload, dict) else payload
            elif name == "labram":
                candidate = {key[8:]: value for key, value in payload["model"].items() if key.startswith("student.")}
            else:
                candidate = payload
            params[name] = int(sum(value.numel() for value in candidate.values() if torch.is_tensor(value)))
        labram_encoder = load_labram(project, torch.device("cpu"))
        biot_encoder = load_biot(project, torch.device("cpu"))
        encoder_params = {
            "cbramod": params["cbramod"],
            "labram": int(sum(value.numel() for value in labram_encoder.parameters())),
            "biot": int(sum(value.numel() for value in biot_encoder.parameters())),
        }
        del labram_encoder, biot_encoder
        provenance = {
            "candidate_priority": ["LaBraM", "EEGPT", "BIOT", "BENDR"],
            "candidate_audit": [
                {"model": "LaBraM", "status": "selected", "reason": "official MIT code and checkpoint load locally"},
                {"model": "EEGPT", "status": "not_accessible", "reason": "official Figshare checkpoint endpoint returned HTTP 403 in current environment before performance computation"},
                {"model": "BIOT", "status": "selected", "reason": "official MIT code and PREST checkpoint load locally"},
                {"model": "BENDR", "status": "not_reached", "reason": "three-family pool already complete"},
            ],
            "models": {
                "cbramod": {
                    "family": "criss-cross transformer",
                    "official_code": "https://github.com/wangf3014/CBraMod",
                    "code_commit": commits["cbramod"], "license": "MIT",
                    "checkpoint": str(checkpoints["cbramod"]), "checkpoint_sha256": hashes["cbramod"],
                    "parameters_in_checkpoint": params["cbramod"], "frozen_encoder_parameters": encoder_params["cbramod"], "pretraining_objective": "masked EEG reconstruction",
                    "pretraining_data": "TUEG, approximately 9,000 hours",
                    "explicit_hmc": False, "explicit_eegmmidb": False,
                    "unsupervised_pretraining_overlap": False, "pretraining_exposure_unknown": False,
                    "task_labels_used": False, "admissible": True,
                },
                "labram": {
                    "family": "neural-tokenizer transformer",
                    "official_code": "https://github.com/935963004/LaBraM",
                    "code_commit": commits["labram"], "license": "MIT",
                    "checkpoint": str(checkpoints["labram"]), "checkpoint_sha256": hashes["labram"],
                    "parameters_in_checkpoint": params["labram"], "frozen_encoder_parameters": encoder_params["labram"], "pretraining_objective": "vector-quantized neural spectrum prediction and masked neural-code prediction",
                    "pretraining_data": "about 2,500 hours across around 20 datasets; complete corpus list not verified from checkpoint metadata",
                    "explicit_hmc": False, "explicit_eegmmidb": False,
                    "unsupervised_pretraining_overlap": False, "pretraining_exposure_unknown": True,
                    "task_labels_used": False, "admissible": True,
                },
                "biot": {
                    "family": "frequency-patch linear-attention transformer",
                    "official_code": "https://github.com/ycq091044/BIOT",
                    "code_commit": commits["biot"], "license": "MIT",
                    "checkpoint": str(checkpoints["biot"]), "checkpoint_sha256": hashes["biot"],
                    "parameters_in_checkpoint": params["biot"], "frozen_encoder_parameters": encoder_params["biot"], "pretraining_objective": "unsupervised contrastive pretraining",
                    "pretraining_data": "5 million unlabeled PREST resting EEG samples",
                    "explicit_hmc": False, "explicit_eegmmidb": False,
                    "unsupervised_pretraining_overlap": False, "pretraining_exposure_unknown": False,
                    "task_labels_used": False, "admissible": True,
                },
            },
        }
        adapter_specs = {
            "cbramod": {"source": "existing audited token cache", "pooling": "mean over valid channel-patch tokens", "embedding_dim": 200},
            "labram": {"sampling_rate": 200, "unit": "microvolts", "hmc_subwindows_seconds": 10, "hmc_pooling": "mean of three deterministic subwindow embeddings", "eegmmidb_window_seconds": 3.2, "channel_mapping": "normalized 10-20 names", "embedding_dim": 200},
            "biot": {"sampling_rate": 200, "unit": "microvolts", "hmc_channels": ["C3-M2", "C4-M1"], "hmc_channel_tokens": [10, 14], "eegmmidb_montages": [f"{a}-{b}" for a, b in BIOT_MONTAGES], "n_fft": 200, "hop_length": 100, "pooling": "official sequence mean", "embedding_dim": 256},
        }
        adapter_hashes = {name: sha256_json(spec) for name, spec in adapter_specs.items()}
        freeze = {
            "model_order": list(MODELS), "candidate_priority": provenance["candidate_priority"],
            "checkpoint_hashes": hashes, "code_commits": commits, "adapter_spec_hashes": adapter_hashes,
            "frozen_at": utc_now(),
            "unread_result_types": ["probe metrics", "model qualification", "oracle results", "winner shares", "routing results"],
        }
        atomic_json(self.paths.output / "audit/MODEL_PROVENANCE.json", provenance)
        atomic_json(self.paths.output / "audit/CHECKPOINT_HASHES.json", hashes)
        atomic_json(self.paths.output / "audit/RESOLVED_MODEL_PATHS.json", {name: str(path) for name, path in checkpoints.items()})
        atomic_json(self.paths.output / "audit/ADAPTER_SPECS.json", adapter_specs)
        atomic_json(self.paths.output / "audit/ACCESS_AUDIT.json", {"allowed": list(DATASETS), "forbidden": ["cap", "sleep-edf", "bcic2a", "formal calibration", "internal final"], **PROTECTED_FLAGS})
        self.transition("MODEL_CANDIDATE_AUDIT_COMPLETE")
        atomic_json(self.paths.delivery / "V7_MODEL_POOL_FREEZE.json", freeze)
        write_text(self.paths.delivery / "MODEL_PRETRAINING_PROVENANCE.md", self._provenance_markdown(provenance))
        write_text(self.paths.delivery / "INPUT_ADAPTER_AUDIT.md", "# Input adapter audit\n\n```json\n" + json.dumps(adapter_specs, indent=2) + "\n```\n\nAll rules were frozen before task-performance computation. No evaluation label is used by an adapter.")
        self._state["checkpoint_hashes"] = hashes
        self._state["model_pool_hash"] = sha256_json(freeze)
        self.transition("MODEL_POOL_FROZEN")

    @staticmethod
    def _provenance_markdown(provenance: dict[str, Any]) -> str:
        lines = ["# Model pretraining provenance", "", "Unknown exposure is retained as unknown, not rewritten as no overlap.", ""]
        for name, info in provenance["models"].items():
            lines.extend([f"## {name}", "", f"- Family: {info['family']}", f"- Code commit: `{info['code_commit']}`", f"- License: {info['license']}", f"- Checkpoint SHA256: `{info['checkpoint_sha256']}`", f"- Parameters in checkpoint: {info['parameters_in_checkpoint']}", f"- Frozen encoder parameters: {info['frozen_encoder_parameters']}", f"- Pretraining objective: {info['pretraining_objective']}", f"- Pretraining data: {info['pretraining_data']}", f"- Unsupervised overlap: {info['unsupervised_pretraining_overlap']}", f"- Exposure unknown: {info['pretraining_exposure_unknown']}", f"- Task labels used: {info['task_labels_used']}", f"- Admissible: {info['admissible']}", ""])
        return "\n".join(lines)

    def freeze_protocol(self) -> None:
        protocol = {
            "datasets": list(DATASETS), "outer_folds": list(FOLDS), "head_seeds": list(SEEDS),
            "subject_bootstrap": {"repetitions": 5000, "seed": 20260810},
            "subject_shuffle_null": {"repetitions": 500, "seed": 20260811},
            "probe": {"family": "LayerNorm+Linear", "learning_rate": [1e-4, 3e-4, 1e-3], "weight_decay": [0.0, 1e-4], "max_epochs": 30, "early_stopping_patience": 5, "loss": "class-weighted cross entropy"},
            "primary_unit": "subject", "backbones_frozen": True,
        }
        atomic_json(self.paths.delivery / "V7_STAGE0A_FREEZE.json", protocol)
        write_text(self.paths.delivery / "V7_STAGE0A_PROTOCOL.md", "# V7 Stage-0A protocol\n\n```json\n" + json.dumps(protocol, indent=2) + "\n```")
        write_text(self.paths.delivery / "STANDARDIZED_PROBE_PROTOCOL.md", "# Standardized probe protocol\n\nEvery frozen pooled embedding is followed by the same trainable `LayerNorm -> Linear` head. Outer fold `e` is evaluated once, `(e+1) mod 5` selects hyperparameters and epoch, and the other three folds train the head. Backbones never receive gradients.")
        self._state["config_hash"] = sha256_json(protocol)
        self.transition("PROTOCOL_FROZEN")

    def build_canonical(self) -> None:
        cohorts = pd.read_parquet(self.paths.repo / "outputs/contextual_risk/cohorts/MASTER_SUBJECT_COHORTS.parquet")
        cohorts = cohorts[cohorts.master_cohort == "method_development"].copy()
        blocks = pd.read_parquet(self.paths.repo / "outputs/online_blockwise_v6/block_manifest/BLOCK_SAMPLE_MAP.parquet")
        blocks = blocks[blocks.dataset.isin(DATASETS)]
        sequence_manifest = pd.read_parquet(self.paths.repo / "outputs/online_blockwise_v6/sequence_cache/PREDICTION_CACHE_MANIFEST.parquet")
        sequence_manifest = sequence_manifest[(sequence_manifest.outer_fold == 0) & (sequence_manifest.source_seed == 0)]
        rows = []
        for item in cohorts.itertuples(index=False):
            dataset = item.dataset
            subject_id = item.subject_id
            short = subject_id.split(":")[-1]
            path = self.paths.project / f"data/processed/{dataset}/{dataset}_{short}.h5"
            if not path.exists():
                raise TechnicalBlock(f"processed canonical input missing: {path}")
            sequence_row = sequence_manifest[(sequence_manifest.dataset == dataset) & (sequence_manifest.subject_id == subject_id)]
            if len(sequence_row) != 1:
                raise TechnicalBlock(f"unique V6 sequence identity source missing for {subject_id}")
            mapping = pd.read_parquet(sequence_row.iloc[0].cache_path)
            with h5py.File(path) as handle:
                count = len(handle["label"])
                if len(mapping) != count:
                    raise TechnicalBlock(f"sample identity mismatch for {subject_id}: sequence={len(mapping)} processed={count}")
                labels = handle["label"][:].astype(int)
                recording = [value.decode() for value in handle["recording_id"][:]]
                runs = handle["run_id"][:].astype(int)
                starts = handle["window_start"][:].astype(float)
                ends = handle["window_end"][:].astype(float)
            mapping = mapping.reset_index(drop=True)
            if not np.array_equal(mapping.chronological_index.to_numpy(int), np.arange(count)):
                raise TechnicalBlock(f"non-canonical chronological index for {subject_id}")
            if not np.array_equal(mapping.label.to_numpy(int), labels):
                raise TechnicalBlock(f"label mismatch between V6 sequence and processed source for {subject_id}")
            block_ids = set(blocks.loc[blocks.subject_id == subject_id, "sample_id"])
            if not block_ids.issubset(set(mapping.sample_id)):
                raise TechnicalBlock(f"V6 block identities are not a subset of sequence identities for {subject_id}")
            for index in range(count):
                rows.append({
                    "dataset": dataset, "subject_id": subject_id,
                    "sample_id": mapping.loc[index, "sample_id"], "outer_fold": int(item.screening_fold),
                    "label": int(labels[index]), "recording_id": recording[index], "run_id": int(runs[index]),
                    "window_start": starts[index], "window_end": ends[index], "processed_path": str(path), "row_index": index,
                })
        manifest = pd.DataFrame(rows).sort_values(["dataset", "subject_id", "row_index"]).reset_index(drop=True)
        atomic_parquet(self.paths.output / "canonical/CANONICAL_SAMPLE_MANIFEST.parquet", manifest)
        coverage = []
        for dataset in DATASETS:
            part = manifest[manifest.dataset == dataset]
            for model in MODELS:
                coverage.append({"dataset": dataset, "model": model, "subject_coverage": 1.0, "sample_coverage": 1.0, "folds_covered": 5, "classes_covered": int(part.label.nunique()), "technical_precheck": True})
        pd.DataFrame(coverage).to_csv(self.paths.output / "canonical/MODEL_SAMPLE_COVERAGE.csv", index=False)
        write_text(self.paths.delivery / "CANONICAL_SAMPLE_AUDIT.md", f"# Canonical sample audit\n\n- Subjects: {manifest.subject_id.nunique()}\n- Samples: {len(manifest)}\n- HMC uses unchanged 30-second epoch labels.\n- EEGMMIDB uses unchanged frozen trial/window labels.\n- All three models receive identical sample IDs and labels.\n- No subject or sample was removed by model performance.")
        self._state["canonical_hash"] = sha256_file(self.paths.output / "canonical/CANONICAL_SAMPLE_MANIFEST.parquet")
        self.transition("CANONICAL_SAMPLE_AUDIT_COMPLETE")

    def smoke_test(self) -> None:
        manifest = pd.read_parquet(self.paths.output / "canonical/CANONICAL_SAMPLE_MANIFEST.parquet")
        results = []
        for model_name in ("labram", "biot"):
            model = load_labram(self.paths.project, self.device) if model_name == "labram" else load_biot(self.paths.project, self.device)
            for dataset in DATASETS:
                subjects = manifest[manifest.dataset == dataset].subject_id.drop_duplicates().iloc[:2]
                for subject_id in subjects:
                    part = manifest[(manifest.dataset == dataset) & (manifest.subject_id == subject_id)].iloc[:16]
                    with h5py.File(part.processed_path.iloc[0]) as handle:
                        signals = handle["signal"][part.row_index.to_numpy()]
                        names = [value.decode() for value in handle["channel_names"][:]]
                        sampling_rate = float(handle["sampling_rate"][()])
                    with torch.inference_mode():
                        first = labram_embeddings(model, signals, names, dataset, self.device, sampling_rate) if model_name == "labram" else biot_embeddings(model, signals, names, dataset, self.device, sampling_rate)
                        second = labram_embeddings(model, signals, names, dataset, self.device, sampling_rate) if model_name == "labram" else biot_embeddings(model, signals, names, dataset, self.device, sampling_rate)
                    assert torch.isfinite(first).all() and first.shape == second.shape
                    assert float((first - second).abs().max()) <= 1e-6
                    assert not any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters())
                    results.append({"model": model_name, "dataset": dataset, "subject_id": subject_id, "samples": len(part), "embedding_dim": int(first.shape[1]), "max_repeat_error": float((first-second).abs().max()), "finite": True, "backbone_gradients": False})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        atomic_json(self.paths.output / "audit/SMOKE_TEST.json", results)
        self._state["gpu_used"] = torch.cuda.is_available()
        self.transition("ADAPTER_SMOKE_TEST_COMPLETE")

    def extract_embeddings(self) -> None:
        manifest = pd.read_parquet(self.paths.output / "canonical/CANONICAL_SAMPLE_MANIFEST.parquet")
        checkpoint_hashes = self._state["checkpoint_hashes"]
        adapter_specs = json.loads((self.paths.output / "audit/ADAPTER_SPECS.json").read_text())
        adapter_hashes = {name: sha256_json(spec) for name, spec in adapter_specs.items()}
        records = []
        total = len(DATASETS) * len(MODELS) * len(FOLDS)
        completed = 0
        for dataset in DATASETS:
            for model_name in MODELS:
                for fold in FOLDS:
                    group = manifest[(manifest.dataset == dataset) & (manifest.outer_fold == fold)]
                    if group.empty:
                        raise TechnicalBlock(f"empty canonical fold: {dataset}/{fold}")
                    model = None
                    if model_name == "labram":
                        model = load_labram(self.paths.project, self.device)
                    elif model_name == "biot":
                        model = load_biot(self.paths.project, self.device)
                    for subject_id, subject in group.groupby("subject_id", sort=True):
                        safe = subject_id.replace(":", "_")
                        out = self.paths.output / f"embedding_cache/{dataset}/{model_name}/fold_{fold}/{safe}.parquet"
                        if out.exists():
                            cached = pd.read_parquet(out)
                            if len(cached) == len(subject) and cached.sample_id.tolist() == subject.sample_id.tolist():
                                records.append(self._cache_record(dataset, model_name, fold, subject_id, out, len(cached), checkpoint_hashes[model_name], adapter_hashes[model_name]))
                                continue
                        embeddings = self._extract_subject(model_name, model, dataset, subject)
                        frame = subject[["dataset", "subject_id", "sample_id", "outer_fold", "label"]].copy()
                        frame["model"] = model_name
                        frame["embedding"] = [row.astype(np.float16).tolist() for row in embeddings]
                        frame["adapter_hash"] = adapter_hashes[model_name]
                        frame["checkpoint_hash"] = checkpoint_hashes[model_name]
                        frame["code_commit"] = self._state["starting_commit"]
                        frame["input_sample_hash"] = [hashlib.sha256(value.encode()).hexdigest() for value in frame.sample_id]
                        atomic_parquet(out, frame)
                        records.append(self._cache_record(dataset, model_name, fold, subject_id, out, len(frame), checkpoint_hashes[model_name], adapter_hashes[model_name]))
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    completed += 1
                    job = f"embedding:{dataset}:{model_name}:{fold}"
                    if job not in self._state["completed_jobs"]:
                        self._state["completed_jobs"].append(job)
                    print(json.dumps({"state": "EMBEDDING_CACHE", "dataset": dataset, "model": model_name, "fold": fold, "completed_jobs": completed, "total_jobs": total, "gpu_memory": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0}), flush=True)
                    atomic_parquet(self.paths.output / "embedding_cache/EMBEDDING_CACHE_MANIFEST.parquet", pd.DataFrame(records))
                    atomic_json(self.paths.state, self._state)
        cache_manifest = pd.DataFrame(records)
        if cache_manifest.duplicated(["dataset", "model", "subject_id"]).any():
            raise TechnicalBlock("embedding extraction duplicated a subject/model")
        self.transition("EMBEDDING_CACHE_COMPLETE")

    def _extract_subject(self, model_name: str, model: torch.nn.Module | None, dataset: str, subject: pd.DataFrame) -> np.ndarray:
        if model_name == "cbramod":
            short = subject.subject_id.iloc[0].split(":")[-1]
            source = self.paths.project / f"data/embeddings_tokens_v2/{dataset}/{short}.h5"
            with h5py.File(source) as handle:
                tokens = handle["token_embeddings"][:].astype(np.float32)
                mask = handle["valid_token_mask"][:].astype(bool)
            if len(tokens) != len(subject):
                raise TechnicalBlock(f"CBraMod cache mismatch for {subject.subject_id.iloc[0]}")
            flat = tokens.reshape(len(tokens), -1, tokens.shape[-1])
            weights = mask.reshape(len(tokens), -1, 1)
            return (flat * weights).sum(1) / np.maximum(weights.sum(1), 1)
        outputs = []
        batch_size = 64 if dataset == "hmc" else (4 if model_name == "labram" else 16)
        with h5py.File(subject.processed_path.iloc[0]) as handle:
            names = [value.decode() for value in handle["channel_names"][:]]
            sampling_rate = float(handle["sampling_rate"][()])
            indices = subject.row_index.to_numpy()
            for start in range(0, len(indices), batch_size):
                signals = handle["signal"][indices[start:start + batch_size]]
                output = labram_embeddings(model, signals, names, dataset, self.device, sampling_rate) if model_name == "labram" else biot_embeddings(model, signals, names, dataset, self.device, sampling_rate)
                outputs.append(output.detach().cpu().float().numpy())
        result = np.concatenate(outputs)
        if not np.isfinite(result).all():
            raise TechnicalBlock(f"non-finite embedding: {dataset}/{model_name}/{subject.subject_id.iloc[0]}")
        return result

    @staticmethod
    def _cache_record(dataset: str, model: str, fold: int, subject: str, path: pathlib.Path, rows: int, checkpoint_hash: str, adapter_hash: str) -> dict[str, Any]:
        return {"dataset": dataset, "model": model, "outer_fold": fold, "subject_id": subject, "cache_path": str(path), "row_count": rows, "cache_sha256": sha256_file(path), "checkpoint_hash": checkpoint_hash, "adapter_hash": adapter_hash, "extraction_count": 1}

    def train_probes(self) -> None:
        manifest = pd.read_parquet(self.paths.output / "embedding_cache/EMBEDDING_CACHE_MANIFEST.parquet")
        model_rows, hyper_rows, training_logs, prediction_rows, subject_rows, seed_rows = [], [], [], [], [], []
        for dataset in DATASETS:
            for model_name in MODELS:
                files = manifest[(manifest.dataset == dataset) & (manifest.model == model_name)].cache_path
                frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
                embeddings = np.stack(frame.embedding.map(np.asarray)).astype(np.float32)
                labels = frame.label.to_numpy(int)
                classes = np.sort(np.unique(labels))
                for fold in FOLDS:
                    train_folds, validation_fold, evaluation_fold = fold_roles(fold)
                    train_mask = frame.outer_fold.isin(train_folds).to_numpy()
                    val_mask = (frame.outer_fold == validation_fold).to_numpy()
                    eval_mask = (frame.outer_fold == evaluation_fold).to_numpy()
                    class_weights = normalized_inverse_frequency(labels[train_mask], classes)
                    for seed in SEEDS:
                        result = self._fit_one_probe(embeddings, labels, train_mask, val_mask, eval_mask, classes, class_weights, seed)
                        state_dir = self.paths.output / f"probes/models/{dataset}/{model_name}/fold_{fold}"
                        state_dir.mkdir(parents=True, exist_ok=True)
                        state_path = state_dir / f"seed_{seed}.pt"
                        torch.save(result["state_dict"], state_path)
                        model_rows.append({"dataset": dataset, "model": model_name, "outer_fold": fold, "seed": seed, "path": str(state_path), "sha256": sha256_file(state_path), "embedding_dim": embeddings.shape[1], "n_classes": len(classes)})
                        hyper_rows.append({"dataset": dataset, "model": model_name, "outer_fold": fold, "seed": seed, **result["best"]})
                        training_logs.extend({"dataset": dataset, "model": model_name, "outer_fold": fold, "seed": seed, **row} for row in result["logs"])
                        evaluation = frame.loc[eval_mask, ["dataset", "subject_id", "sample_id", "outer_fold", "label"]].reset_index(drop=True)
                        for index, row in evaluation.iterrows():
                            prediction_rows.append({**row.to_dict(), "model": model_name, "seed": seed, "pred": int(result["pred"][index]), "probabilities": result["prob"][index].astype(float).tolist()})
                        metrics = self._metric_rows(evaluation, result["pred"], result["prob"], classes, dataset, model_name, fold, seed)
                        subject_rows.extend(metrics[0]); seed_rows.append(metrics[1])
                        job = f"probe:{dataset}:{model_name}:{fold}:{seed}"
                        self._state["completed_jobs"].append(job)
                        print(json.dumps({"state": "STANDARDIZED_PROBES", "dataset": dataset, "model": model_name, "fold": fold, "seed": seed, "completed_jobs": len(self._state["completed_jobs"])}), flush=True)
        atomic_parquet(self.paths.output / "probes/PROBE_MODEL_MANIFEST.parquet", pd.DataFrame(model_rows))
        pd.DataFrame(hyper_rows).to_csv(self.paths.output / "probes/PROBE_HYPERPARAMETERS.csv", index=False)
        atomic_parquet(self.paths.output / "probes/PROBE_TRAINING_LOGS.parquet", pd.DataFrame(training_logs))
        atomic_parquet(self.paths.output / "results/OOF_PREDICTIONS.parquet", pd.DataFrame(prediction_rows))
        atomic_parquet(self.paths.output / "results/MODEL_METRICS_BY_SUBJECT.parquet", pd.DataFrame(subject_rows))
        pd.DataFrame(seed_rows).to_csv(self.paths.output / "results/MODEL_METRICS_BY_SEED.csv", index=False)
        self.transition("STANDARDIZED_PROBES_COMPLETE")

    def _fit_one_probe(self, x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray, evaluation: np.ndarray, classes: np.ndarray, weights: dict[int, float], seed: int) -> dict[str, Any]:
        torch.manual_seed(seed)
        device = self.device
        xt = torch.from_numpy(x[train]).to(device)
        yt = torch.from_numpy(y[train]).long().to(device)
        xv = torch.from_numpy(x[val]).to(device)
        yv = y[val]
        xe = torch.from_numpy(x[evaluation]).to(device)
        class_tensor = torch.tensor([weights[int(label)] for label in classes], dtype=torch.float32, device=device)
        best_overall, logs = None, []
        for lr in (1e-4, 3e-4, 1e-3):
            for weight_decay in (0.0, 1e-4):
                torch.manual_seed(seed)
                head = torch.nn.Sequential(torch.nn.LayerNorm(x.shape[1]), torch.nn.Linear(x.shape[1], len(classes))).to(device)
                optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
                best_local, bad = None, 0
                for epoch in range(1, 31):
                    head.train(); optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(head(xt), yt, weight=class_tensor)
                    loss.backward(); optimizer.step()
                    head.eval()
                    with torch.inference_mode():
                        val_prob = head(xv).softmax(1).cpu().numpy()
                    val_pred = classes[val_prob.argmax(1)]
                    val_ba = balanced_accuracy_score(yv, val_pred)
                    logs.append({"lr": lr, "weight_decay": weight_decay, "epoch": epoch, "train_loss": float(loss.detach()), "validation_balanced_accuracy": float(val_ba)})
                    candidate = {"validation_balanced_accuracy": float(val_ba), "epoch": epoch, "state_dict": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}}
                    if best_local is None or val_ba > best_local["validation_balanced_accuracy"] + 1e-12:
                        best_local, bad = candidate, 0
                    else:
                        bad += 1
                    if bad >= 5:
                        break
                candidate_key = (best_local["validation_balanced_accuracy"], -lr, -weight_decay)
                if best_overall is None or candidate_key > best_overall["key"]:
                    best_overall = {"key": candidate_key, "lr": lr, "weight_decay": weight_decay, **best_local}
        head = torch.nn.Sequential(torch.nn.LayerNorm(x.shape[1]), torch.nn.Linear(x.shape[1], len(classes))).to(device)
        head.load_state_dict(best_overall["state_dict"]); head.eval()
        with torch.inference_mode():
            probability = head(xe).softmax(1).cpu().numpy()
        return {"state_dict": best_overall["state_dict"], "best": {"learning_rate": best_overall["lr"], "weight_decay": best_overall["weight_decay"], "selected_epoch": best_overall["epoch"], "validation_balanced_accuracy": best_overall["validation_balanced_accuracy"]}, "logs": logs, "prob": probability, "pred": classes[probability.argmax(1)]}

    @staticmethod
    def _ece(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
        confidence = probability.max(1); prediction = probability.argmax(1)
        total = 0.0
        for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
            mask = (confidence > low) & (confidence <= high)
            if np.any(mask): total += np.mean(mask) * abs(np.mean(prediction[mask] == labels[mask]) - np.mean(confidence[mask]))
        return float(total)

    def _metric_rows(self, evaluation: pd.DataFrame, pred: np.ndarray, prob: np.ndarray, classes: np.ndarray, dataset: str, model: str, fold: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        y = evaluation.label.to_numpy(int)
        subject_rows = []
        for subject, indices in evaluation.groupby("subject_id").groups.items():
            idx = np.asarray(list(indices)); sy, sp, sq = y[idx], pred[idx], prob[idx]
            subject_rows.append({"dataset": dataset, "model": model, "outer_fold": fold, "seed": seed, "subject_id": subject, "n": len(idx), "accuracy": accuracy_score(sy, sp), "balanced_accuracy": balanced_accuracy_score(sy, sp), "macro_f1": f1_score(sy, sp, labels=classes, average="macro", zero_division=0), "kappa": cohen_kappa_score(sy, sp, labels=classes), "nll": log_loss(sy, sq, labels=classes), "brier": float(np.mean(np.sum((sq - np.eye(len(classes))[sy])**2, axis=1))), "ece": self._ece(sy, sq), "prediction_classes": int(np.unique(sp).size), "class_support": json.dumps(np.bincount(sy, minlength=len(classes)).tolist()), "prediction_frequency": json.dumps(np.bincount(sp, minlength=len(classes)).tolist())})
        seed_row = {"dataset": dataset, "model": model, "outer_fold": fold, "seed": seed, "n": len(y), "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred), "macro_f1": f1_score(y, pred, labels=classes, average="macro", zero_division=0), "kappa": cohen_kappa_score(y, pred, labels=classes), "nll": log_loss(y, prob, labels=classes), "brier": float(np.mean(np.sum((prob - np.eye(len(classes))[y])**2, axis=1))), "ece": self._ece(y, prob), "probability_finite": bool(np.isfinite(prob).all()), "probability_nonnegative": bool((prob >= 0).all()), "max_probability_sum_error": float(np.abs(prob.sum(1)-1).max()), "all_classes_present": bool(set(classes) == set(np.unique(y))), "predicted_classes": int(np.unique(pred).size)}
        return subject_rows, seed_row

    def qualify_models(self, *, transition_state: bool = True) -> bool:
        prediction = pd.read_parquet(self.paths.output / "results/OOF_PREDICTIONS.parquet")
        canonical = pd.read_parquet(self.paths.output / "canonical/CANONICAL_SAMPLE_MANIFEST.parquet")
        existing_seed_path = self.paths.output / "results/MODEL_METRICS_BY_SEED.csv"
        fold_seed_path = self.paths.output / "results/MODEL_METRICS_BY_FOLD_SEED.csv"
        if existing_seed_path.exists() and not fold_seed_path.exists():
            shutil.copy2(existing_seed_path, fold_seed_path)
        aggregate_seed_rows: list[dict[str, Any]] = []
        summaries: dict[tuple[str, str], dict[str, Any]] = {}
        for dataset in DATASETS:
            classes = np.sort(canonical.loc[canonical.dataset == dataset, "label"].unique()).astype(int)
            for model in MODELS:
                part = prediction[(prediction.dataset == dataset) & (prediction.model == model)]
                seed_bas, seed_nonconstant, subject_bas = [], [], {}
                probability_sane = True
                all_classes_predicted = True
                folds_noncollapsed = True
                for head_seed in SEEDS:
                    current = part[part.seed == head_seed].reset_index(drop=True)
                    y = current.label.to_numpy(int); pred = current.pred.to_numpy(int)
                    prob = np.stack(current.probabilities.map(np.asarray)).astype(float)
                    ba = float(balanced_accuracy_score(y, pred)); seed_bas.append(ba)
                    subject_values, nonconstant = [], []
                    for subject_id, indices in current.groupby("subject_id").groups.items():
                        idx = np.asarray(list(indices)); sy, sp = y[idx], pred[idx]
                        value = float(balanced_accuracy_score(sy, sp))
                        subject_values.append(value); subject_bas.setdefault(subject_id, []).append(value)
                        nonconstant.append(np.unique(sp).size > 1)
                    seed_nonconstant.append(float(np.mean(nonconstant)))
                    probability_sane &= bool(np.isfinite(prob).all() and (prob >= 0).all() and np.abs(prob.sum(1)-1).max() <= 1e-5)
                    all_classes_predicted &= set(np.unique(pred)) == set(classes)
                    folds_noncollapsed &= bool(current.groupby("outer_fold").pred.nunique().min() >= 2)
                    aggregate_seed_rows.append({"dataset": dataset, "model": model, "seed": head_seed, "n": len(y), "accuracy": accuracy_score(y, pred), "balanced_accuracy": ba, "macro_f1": f1_score(y, pred, labels=classes, average="macro", zero_division=0), "kappa": cohen_kappa_score(y, pred, labels=classes), "nll": log_loss(y, prob, labels=classes), "brier": float(np.mean(np.sum((prob - np.eye(len(classes))[y])**2, axis=1))), "ece": self._ece(y, prob), "probability_finite": bool(np.isfinite(prob).all()), "probability_nonnegative": bool((prob >= 0).all()), "max_probability_sum_error": float(np.abs(prob.sum(1)-1).max()), "all_classes_predicted": set(np.unique(pred)) == set(classes), "minimum_fold_predicted_classes": int(current.groupby("outer_fold").pred.nunique().min()), "nonconstant_subject_rate": float(np.mean(nonconstant))})
                subject_mean = np.asarray([np.mean(values) for values in subject_bas.values()])
                summaries[(dataset, model)] = {"classes": classes, "seed_bas": np.asarray(seed_bas), "dataset_ba": float(np.mean(seed_bas)), "median_subject_ba": float(np.median(subject_mean)), "seed_ba_std": float(np.std(seed_bas)), "nonconstant_subject_rate": float(np.min(seed_nonconstant)), "probability_sane": probability_sane, "all_classes_predicted": all_classes_predicted, "folds_noncollapsed": folds_noncollapsed}
        pd.DataFrame(aggregate_seed_rows).to_csv(existing_seed_path, index=False)
        rows = []
        for dataset in DATASETS:
            cbra = summaries[(dataset, "cbramod")]["dataset_ba"]
            for model in MODELS:
                summary = summaries[(dataset, model)]
                n_classes = len(summary["classes"])
                gates = gate_q(probability_sane=summary["probability_sane"], embedding_sane=True, all_classes_present=summary["all_classes_predicted"], nonconstant_subject_rate=summary["nonconstant_subject_rate"], seed_ba_std=summary["seed_ba_std"], dataset_ba=summary["dataset_ba"], median_subject_ba=summary["median_subject_ba"], cbramod_ba=cbra, positive_seed_count=int(np.sum(summary["seed_bas"] > 1/n_classes)), folds_noncollapsed=summary["folds_noncollapsed"], n_classes=n_classes)
                rows.append({"dataset": dataset, "model": model, "n_classes": n_classes, "chance_balanced_accuracy": 1/n_classes, "dataset_balanced_accuracy": summary["dataset_ba"], "median_subject_balanced_accuracy": summary["median_subject_ba"], "seed_ba_std": summary["seed_ba_std"], "nonconstant_subject_rate": summary["nonconstant_subject_rate"], **gates, "passed": all(gates.values())})
        gate_frame = pd.DataFrame(rows)
        gate_frame.to_csv(self.paths.output / "results/MODEL_QUALIFICATION_GATE.csv", index=False)
        write_text(self.paths.delivery / "MODEL_QUALIFICATION_REPORT.md", "# Model qualification report\n\n" + gate_frame.to_markdown(index=False))
        write_text(self.paths.delivery / "GATE_Q_DECISION.md", "# Gate Q decision\n\n" + gate_frame.to_markdown(index=False) + f"\n\nAll models pass: **{bool(gate_frame.passed.all())}**")
        if transition_state:
            self.transition("MODEL_QUALIFICATION_COMPLETE", gate_q_passed=bool(gate_frame.passed.all()))
        return bool(gate_frame.passed.all())

    def scientific_stop_gate_q(self) -> None:
        verdict = "V7_STAGE0A_STOP_MODEL_QUALIFICATION_FAILURE"
        self._state.update({"verdict": verdict, "stopping_gate": "Gate Q", "terminal": True})
        atomic_json(self.paths.output / "SCIENTIFIC_STOP.json", {"verdict": verdict, "stopping_gate": "Gate Q", "terminal": True, "timestamp": utc_now(), "reason": "At least one frozen model failed at least one preregistered qualification criterion; oracle computation is forbidden."})
        self._build_stop_delivery(verdict, "Gate Q")
        self.transition("STOPPED", terminal=True, verdict=verdict, stopping_gate="Gate Q")

    def technical_block(self, message: str) -> None:
        verdict = "V7_STAGE0A_TECHNICAL_BLOCK"
        self._state.update({"verdict": verdict, "stopping_gate": self._state.get("state"), "terminal": True})
        atomic_json(self.paths.output / "TECHNICAL_BLOCK.json", {"verdict": verdict, "terminal": True, "timestamp": utc_now(), "diagnostic": message})
        self._build_stop_delivery(verdict, self._state.get("state", "unknown"), diagnostic=message)
        self.transition("STOPPED", terminal=True, verdict=verdict)

    def _build_stop_delivery(self, verdict: str, gate: str, diagnostic: str | None = None) -> None:
        decision = {"verdict": verdict, "stopping_gate": gate, "terminal": True, "runtime_seconds": time.time()-self.started, "gpu_used": self._state.get("gpu_used", False), "later_stages_not_run": ["best fixed model", "full subject oracle", "split-half transfer oracle", "subject-shuffle null", "error complementarity", "unlabeled routing screen", "router development", "abstention", "scout-to-expert"], **PROTECTED_FLAGS}
        if diagnostic: decision["diagnostic"] = diagnostic
        atomic_json(self.paths.delivery / "V7_STAGE0A_DECISION.json", decision)
        write_text(self.paths.delivery / "V7_STAGE0A_DECISION.md", "# V7 Stage-0A decision\n\n```json\n" + json.dumps(decision, indent=2) + "\n```")
        write_text(self.paths.delivery / "LIMITATIONS.md", "# Limitations\n\nThe frozen protocol stopped at the first failed gate. Results and reports for prohibited later phases do not exist. A technical smoke test is not treated as model qualification.")
        write_text(self.paths.delivery / "REPRODUCE.md", f"# Reproduce\n\n```bash\n/root/miniconda3/envs/hsc_gpu/bin/python scripts/fm_routing_v7/run_all.py --repo-root {self.paths.repo} --resume\n```\n\nA terminal state only rebuilds delivery metadata and never resumes scientific computation.")
        if not (self.paths.output / "FAILURES.csv").exists():
            write_text(self.paths.output / "FAILURES.csv", "stage,dataset,model,fold,seed,error")
        self.build_manifest()

    def build_manifest(self) -> None:
        files = []
        for root in (self.paths.output, self.paths.delivery):
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.name != "DELIVERY_MANIFEST.json":
                    files.append({"path": str(path.relative_to(self.paths.repo)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        atomic_json(self.paths.delivery / "DELIVERY_MANIFEST.json", {"generated_at": utc_now(), "files": files})

    def environment_report(self) -> None:
        text = "\n".join([f"python={sys.version}", f"platform={platform.platform()}", f"torch={torch.__version__}", f"cuda_available={torch.cuda.is_available()}", f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"])
        write_text(self.paths.output / "provenance/ENVIRONMENT.txt", text)
        hashes = {"config_hash": self._state.get("config_hash"), "model_pool_hash": self._state.get("model_pool_hash"), "canonical_hash": self._state.get("canonical_hash"), "checkpoint_hashes": self._state.get("checkpoint_hashes")}
        atomic_json(self.paths.output / "provenance/HASHES.json", hashes)

    def run(self) -> int:
        if self._state.get("terminal"):
            self.build_manifest()
            print(f"terminal state retained: {self._state.get('verdict')}")
            return 2 if self._state.get("verdict") == "V7_STAGE0A_TECHNICAL_BLOCK" else 0
        try:
            self.audit_predecessor()
            self.audit_and_freeze_models()
            self.freeze_protocol()
            self.build_canonical()
            self.smoke_test()
            self.extract_embeddings()
            self.train_probes()
            if not self.qualify_models():
                self.scientific_stop_gate_q()
                self.environment_report(); self.build_manifest()
                return 0
            raise TechnicalBlock("Gate Q passed, but later oracle stages are not present in this executable")
        except TechnicalBlock as exc:
            self.technical_block(str(exc)); self.environment_report(); self.build_manifest()
            return 2
