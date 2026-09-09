"""Outcome-blind source and TeCh-fidelity preflight for the seven-backbone benchmark.

This entry point intentionally never opens a cache, split outcome, prediction,
or held-out artifact.  It only fingerprints source/checkpoint availability and
tests the TeCh wrapper on synthetic tensors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from tech_official import TECH_COMMIT, TeChAdapter, official_model, official_root, recipe_config


EXP = Path(__file__).resolve().parents[1]
PROTOCOL = EXP / "protocol"
EXPECTED = {
    "TeCh": ("Levi-Ackman/TeCh", TECH_COMMIT, "TeCh"),
    "TCFormer": ("Altaheri/TCFormer", "701a361640708ef1cc07cf22660d47edc6d9bf3c", "TCFormer"),
    "ST-EEGFormer-small": ("LiuyinYang1101/STEEGFormer", "542ee17918c3c2c36ba1d4ea02bedff5eb149370", "STEEGFormer"),
    "LaBraM-base": ("935963004/LaBraM", "c431221e6cfd23dbfa9950e0180682fb322b0548", "LaBraM"),
    "CBraMod": ("wjq-learning/CBraMod", "b9e961003214326972c567eff390e75b0287e32a", "CBraMod"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def find_checkpoint(root: Path, model: str) -> Path | None:
    patterns = {
        "ST-EEGFormer-small": ("*small*.pth", "*small*.pt", "checkpoint-300.pth"),
        "LaBraM-base": ("*labram*base*.pth", "*labram*base*.pt"),
        "CBraMod": ("*cbramod*.pth", "*pretrained*.pth"),
    }.get(model, ())
    for pattern in patterns:
        found = sorted(root.rglob(pattern))
        if found:
            return found[0]
    return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_tech_fidelity() -> dict[str, Any]:
    torch.manual_seed(7001)
    config = recipe_config(channels=62, samples=250, classes=4, recipe="TECH-T")
    direct = official_model(config).eval()
    wrapped = TeChAdapter(config).eval()
    wrapped.model.load_state_dict(direct.state_dict(), strict=True)
    x = torch.randn(3, 62, 250)
    with torch.no_grad():
        direct_logits = direct(x.transpose(1, 2).contiguous())
        wrapped_logits_a = wrapped(x)
        wrapped_logits_b = wrapped(x)
    delta = float((direct_logits - wrapped_logits_a).abs().max().item())
    deterministic_delta = float((wrapped_logits_a - wrapped_logits_b).abs().max().item())
    return {
        "official_repository": "Levi-Ackman/TeCh",
        "official_commit": TECH_COMMIT,
        "source_sha256": sha256(official_root() / "models" / "TeCh.py"),
        "adapter": "B,C,T -> contiguous transpose(1,2) -> official Model",
        "synthetic_input_shape": [3, 62, 250],
        "max_abs_logit_diff": delta,
        "eval_repeat_max_abs_diff": deterministic_delta,
        "parameter_count": int(sum(p.numel() for p in wrapped.parameters())),
        "pass": bool(delta <= 1e-6 and deterministic_delta <= 1e-6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default=os.environ.get("OFFICIAL_BACKBONE_ROOT", ""))
    parser.add_argument("--fm-root", default=os.environ.get("FM_RESCUE_RUNTIME", ""))
    args = parser.parse_args()
    if not args.official_root:
        raise RuntimeError("OFFICIAL_BACKBONE_ROOT must point to the separately managed official source directory")
    official = Path(args.official_root).resolve()
    fm_root = Path(args.fm_root).resolve() if args.fm_root else None
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    fidelity = run_tech_fidelity()
    write_json(PROTOCOL / "TECH_WRAPPER_FIDELITY.json", fidelity)
    rows: list[dict[str, Any]] = []
    # EEGNet/LiteBN are project-canonical, not cloned external backbones.
    canonical = Path(__file__).resolve().parents[3] / "experiments" / "persist_eeg_carrier_dualdataset_screen_v1" / "code"
    for model, source in (("EEGNet", "CRCICLR canonical locked implementation"), ("LiteBN", "CRCICLR CompactLite BN implementation")):
        rows.append({"model": model, "official_repository_or_source": source, "source_commit": None,
                     "source_present": canonical.is_dir(), "pretrained": "no", "checkpoint_path": None,
                     "checkpoint_sha256": None, "wrapper_source": "benchmark_core.py (pending exact-reuse audit)",
                     "trainable_parameter_count": None, "input_sampling_assumption": "frozen cache native rate",
                     "channel_mapping": "identity", "input_adapter": "none", "recipe_source": "historical frozen CRCICLR recipe",
                     "known_deviation_from_official": "none claimed; reuse ledger required before use"})
    for model, (source, expected_commit, directory) in EXPECTED.items():
        path = official / directory
        actual = git_commit(path)
        checkpoint = None
        if model != "TeCh":
            checkpoint = find_checkpoint(path, model)
            if checkpoint is None and fm_root is not None and fm_root.exists():
                checkpoint = find_checkpoint(fm_root, model)
        rows.append({"model": model, "official_repository_or_source": source, "source_commit": actual,
                     "expected_source_commit": expected_commit, "source_present": path.is_dir(),
                     "source_commit_matches": actual == expected_commit, "pretrained": "no" if model in ("TeCh", "TCFormer") else "required",
                     "checkpoint_path": None if checkpoint is None else str(checkpoint),
                     "checkpoint_sha256": None if checkpoint is None else sha256(checkpoint),
                     "wrapper_source": "tech_official.py" if model == "TeCh" else "not yet admitted; explicit adapter audit pending",
                     "trainable_parameter_count": fidelity["parameter_count"] if model == "TeCh" else None,
                     "input_sampling_assumption": "native frozen cache" if model == "TeCh" else "pending official adapter audit",
                     "channel_mapping": "identity" if model == "TeCh" else "pending official adapter audit",
                     "input_adapter": "B,C,T -> B,T,C only" if model == "TeCh" else "pending official adapter audit",
                     "recipe_source": "official TDBRAIN/APAVA preregistered selection" if model == "TeCh" else "pending source audit",
                     "known_deviation_from_official": "none" if model == "TeCh" else "not admitted until audit"})
    columns = sorted({key for row in rows for key in row})
    with (PROTOCOL / "BACKBONE_SOURCE_PROVENANCE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    write_json(PROTOCOL / "PREFLIGHT_NO_OUTCOME_AUDIT.json", {
        "pass": bool(fidelity["pass"]),
        "actions": ["official source revision checks", "checkpoint availability/hash checks", "synthetic TeCh wrapper equivalence", "synthetic TeCh eval determinism"],
        "explicitly_not_accessed": ["EEG cache tensors", "SEARCH outer-dev predictions", "fixed-held-out data", "performance/outcome artifacts"],
        "tech_fidelity": fidelity,
        "next_gate": "No inner training or outer prediction is permitted until all source/adapters are admitted and recipe selection is locked.",
    })
    print(json.dumps({"tech_fidelity_pass": fidelity["pass"], "provenance": str(PROTOCOL / "BACKBONE_SOURCE_PROVENANCE.csv")}, sort_keys=True))
    return 0 if fidelity["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
