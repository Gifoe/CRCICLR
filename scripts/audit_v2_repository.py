#!/usr/bin/env python
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import torch


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")
REPO = ROOT / "repo"
OUT = ROOT / "outputs" / "v2_joint_certified" / "provenance"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_files = sorted((ROOT / "data" / "manifests").glob("*"))
    split_files = sorted((ROOT / "data" / "splits").glob("*/*.json"))
    internal_files = sorted((ROOT / "data" / "splits_internal").glob("*/*.json"))
    episode_files = sorted((ROOT / "data" / "episodes_main120").glob("*/*.parquet"))
    checkpoint = ROOT / "checkpoints" / "cbramod" / "pretrained_weights.pth"
    git_sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    payload = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_v1_commit": "28fe62593a30833acd6b925317d6645ed6c15a04",
        "audit_code_commit": git_sha,
        "branch": subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip(),
        "data_manifest_hashes": {str(path): sha256(path) for path in manifest_files},
        "split_hashes": {str(path): sha256(path) for path in split_files + internal_files},
        "episode_hashes": {str(path): sha256(path) for path in episode_files},
        "checkpoint": {"path": str(checkpoint), "size": checkpoint.stat().st_size, "sha256": sha256(checkpoint)},
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "pip_freeze": subprocess.check_output([str(Path(platform.python_implementation()) if False else Path("/root/miniconda3/envs/hsc_gpu/bin/python")), "-m", "pip", "freeze"], text=True).splitlines(),
        },
        "protected_paths": [
            str(ROOT / "outputs" / "full_experiment"), str(ROOT / "data" / "episodes_main120"),
            str(ROOT / "data" / "splits_internal"), str(ROOT / "outputs" / "full_experiment" / "embeddings"),
        ],
        "v2_output_root": str(ROOT / "outputs" / "v2_joint_certified"),
    }
    (OUT / "ENVIRONMENT_V2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"git_sha": git_sha, "manifests": len(manifest_files), "splits": len(split_files + internal_files),
                      "episodes": len(episode_files), "checkpoint_sha256": payload["checkpoint"]["sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
