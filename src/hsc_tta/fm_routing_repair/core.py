from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
from typing import Any, Iterable

import numpy as np
import torch


OLD_READ_ONLY_PREFIXES = (
    "outputs/fm_routing_v7/",
    "delivery/fm_routing_v7/",
    "outputs/online_blockwise_v6/",
    "delivery/online_blockwise_v6/",
)
PROTECTED_FLAGS = (
    "formal_calibration_opened",
    "internal_final_opened",
    "cap_opened",
    "sleep_edf_opened",
    "bcic2a_opened",
    "router_developed",
    "abstention_developed",
    "scout_developed",
    "full_method_entered",
)


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


def guarded_target(repo: pathlib.Path, path: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve()
    relative = resolved.relative_to(repo.resolve()).as_posix()
    if any(relative.startswith(prefix) for prefix in OLD_READ_ONLY_PREFIXES):
        raise PermissionError(f"predecessor path is read-only: {relative}")
    return resolved


def deterministic_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def directory_hashes(root: pathlib.Path, prefixes: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for prefix in prefixes:
        base = root / prefix
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def adapter_gate(model_audits: dict[str, dict[str, Any]], coverage: dict[str, Any], smoke: dict[str, Any]) -> dict[str, bool]:
    return {
        "F1": all(bool(item["official_fidelity"]) for item in model_audits.values()),
        "F2": all(not bool(item["performance_driven_selection"]) for item in model_audits.values()),
        "F3": float(coverage["minimum_subject_coverage"]) == 1.0,
        "F4": float(coverage["minimum_sample_coverage"]) >= 0.95,
        "F5": bool(coverage["sample_id_and_label_exact_match"]),
        "F6": bool(smoke["all_tokens_finite"]),
        "F7": bool(smoke["identity_and_order_recoverable"]),
    }


def all_protected_false(state: dict[str, Any]) -> bool:
    return all(state.get(flag) is False for flag in PROTECTED_FLAGS)
