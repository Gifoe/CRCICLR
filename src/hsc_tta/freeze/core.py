from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, object]) -> str:
    content = {key: value for key, value in payload.items() if key != "manifest_hash"}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def create_freeze_manifest(
    files: Mapping[str, str | Path],
    *,
    git_commit: str,
    metadata: Mapping[str, object],
    output_path: str | Path,
) -> dict[str, object]:
    if not files:
        raise ValueError("at least one frozen file is required")
    entries: dict[str, dict[str, str]] = {}
    for name, raw_path in sorted(files.items()):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[name] = {"path": str(path), "sha256": file_sha256(path)}
    payload: dict[str, object] = {
        "freeze_version": "hsc-critical-index-v1",
        "git_commit": git_commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": entries,
        "metadata": dict(metadata),
    }
    payload["manifest_hash"] = _canonical_hash(payload)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload


def verify_freeze_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise RuntimeError("experiment is not frozen: manifest is missing")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("manifest_hash") != _canonical_hash(payload):
        raise RuntimeError("freeze manifest hash mismatch")
    for name, entry in payload.get("files", {}).items():
        frozen_path = Path(entry["path"])
        if not frozen_path.is_file():
            raise RuntimeError(f"frozen file is missing: {name}")
        if file_sha256(frozen_path) != entry["sha256"]:
            raise RuntimeError(f"freeze hash mismatch: {name}")
    return payload
