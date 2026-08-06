from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any


READ_ONLY_PREFIXES = (
    "outputs/fm_routing_v7/",
    "delivery/fm_routing_v7/",
    "outputs/fm_routing_v7_repair/",
    "delivery/fm_routing_v7_repair/",
    "outputs/online_blockwise_v6/",
    "delivery/online_blockwise_v6/",
)


class ScientificStop(RuntimeError):
    def __init__(self, verdict: str, gate: str, reason: str):
        super().__init__(reason)
        self.verdict = verdict
        self.gate = gate
        self.reason = reason


class TechnicalBlock(RuntimeError):
    pass


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n")
    os.replace(temporary, path)


def guarded_target(repo: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    relative = target.resolve().relative_to(repo.resolve()).as_posix()
    if any(relative.startswith(prefix) for prefix in READ_ONLY_PREFIXES):
        raise PermissionError(f"historical output is read-only: {relative}")
    return target
