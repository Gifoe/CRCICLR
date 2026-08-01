#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone

import torch


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")
OUT = ROOT / "outputs" / "full_experiment" / "environment"
CHECKPOINT = ROOT / "checkpoints" / "cbramod" / "pretrained_weights.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    props = torch.cuda.get_device_properties(0)
    sm = torch.cuda.get_device_capability(0)
    if sm < (12, 0):
        raise RuntimeError(f"unexpected GPU capability: {sm}")
    checkpoint_hash = sha256(CHECKPOINT)
    if checkpoint_hash != "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178":
        raise RuntimeError("official checkpoint hash mismatch")
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": True,
        "gpu": props.name, "capability": list(sm), "total_vram_bytes": props.total_memory,
        "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
        "pytorch_source": "https://download.pytorch.org/whl/cu128",
        "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        "disk": subprocess.check_output(["df", "-h", str(ROOT)], text=True),
        "memory": subprocess.check_output(["free", "-h"], text=True),
        "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip() if Path("/sys/fs/cgroup/memory.max").exists() else "unknown",
    }
    (OUT / "gpu_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    checkpoint = {
        "repository": "https://huggingface.co/weighting666/CBraMod",
        "revision": "500543c7e30bda1b22bfd51a49301b238dee21fd",
        "url": "https://huggingface.co/weighting666/CBraMod/resolve/500543c7e30bda1b22bfd51a49301b238dee21fd/pretrained_weights.pth",
        "filename": CHECKPOINT.name, "size_bytes": CHECKPOINT.stat().st_size,
        "sha256": checkpoint_hash, "license_on_model_card": "Apache-2.0",
    }
    (OUT / "checkpoint_manifest.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    backbone = {
        "official_repository_url": "https://github.com/wjq-learning/CBraMod",
        "repository_commit_sha": git_sha(ROOT / "external" / "CBraMod"), "repository_license": "MIT",
        "checkpoint": checkpoint, "official_input_shape": "batch x channels x patches x 200",
        "sampling_rate_hz": 200, "normalization": "official downstream loaders: microvolts / 100",
        "mask_handling": "optional patch mask; missing patches replaced by fixed zero mask encoding",
        "channel_handling": "variable channel dimension; ACPE convolution, no fixed channel ID lookup",
        "pretraining_dataset": "Temple University Hospital EEG Corpus (TUEG) only",
        "embedding_dimension": 200, "output_definition": "one 200-D representation per channel-patch token",
        "experiment_pooling": "fixed arithmetic mean over all valid channel-patch tokens",
    }
    (OUT / "backbone_revision.json").write_text(json.dumps(backbone, indent=2), encoding="utf-8")
    t3a = {
        "official_repository_url": "https://github.com/matsuolab/T3A",
        "repository_commit_sha": git_sha(ROOT / "external" / "T3A"), "license": "MIT",
        "support_initialization": "final classifier weight rows",
        "entropy_filtering": "per-class ascending softmax entropy",
        "support_budget": "filter_K per predicted class; official default -1, experiment grid -1/10/20",
        "prototype_computation": "L2-normalized supports transposed times one-hot labels, column normalized",
        "inference_rule": "hidden representation dot normalized prototype matrix",
        "experiment_space": "256-D task-head penultimate hidden representation",
    }
    (OUT / "t3a_revision.json").write_text(json.dumps(t3a, indent=2), encoding="utf-8")
    audit = """# CBraMod pretraining-overlap audit

The official CBraMod ICLR 2025 paper, repository README, and pretraining preprocessing code identify TUEG as the sole pretraining corpus. The released checkpoint is the repository-linked author checkpoint. HMC Sleep Staging, CAP Sleep Database, and EEGMMIDB are not TUEG components; EEGMMIDB appears only as a downstream PhysioNet-MI task in the official code/paper. Therefore no direct target-subject/data pretraining overlap was identified. This conclusion is about the documented released checkpoint; the checkpoint file itself does not encode an auditable list of individual pretraining subject IDs, so subject-level provenance cannot be independently reconstructed from weights alone.

Evidence: official paper (ICLR 2025/OpenReview), official README, `preprocessing_tueg_for_pretraining.py`, `preprocessing_physio.py`, and the author Hugging Face model card/revision recorded in `backbone_revision.json`.
"""
    (OUT / "pretraining_overlap_audit.md").write_text(audit, encoding="utf-8")
    python = Path(__import__("sys").executable)
    (OUT / "requirements-gpu.txt").write_text(
        subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True), encoding="utf-8")
    (OUT / "environment-gpu.yml").write_text(
        subprocess.check_output(["/root/miniconda3/bin/conda", "env", "export", "-n", "hsc_gpu"], text=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
