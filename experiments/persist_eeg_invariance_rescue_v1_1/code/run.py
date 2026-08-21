from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common import CONFIG_PATH, EXPERIMENT_ROOT, OUTPUTS, ensure_directories, git_sha, load_config, sha256_file, write_csv, write_json
from data import load_development_split, load_manifest, persist_split_manifests
from functional import analyze_all
from models import build_model
from rescue import run_eligible_rescues
from spectrum import audit_all
from statistics import finalize
from train import run_full, run_smoke


def _device(require_cuda: bool) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda": raise RuntimeError("full V1.1 training is authorized only on the CUDA server")
    return device


def phase0() -> dict[str, Any]:
    ensure_directories(); config = load_config(); rows = []
    for fold in map(int, config["development_folds"]):
        split = load_development_split(fold); manifest = load_manifest(split, split.model_fit_subjects); rows.append({"fold": fold, "model_fit_subjects": len(split.model_fit_subjects), "outcome_subjects": len(split.outcome_subjects), "materialized_rows": len(manifest), "sessions": sorted(map(int, manifest.session_id.unique())), "channels": sorted(map(int, manifest.n_channels.unique())), "sampling_rates": sorted(map(float, manifest.sampling_rate.unique())), "trials_per_subject_session_class": sorted(map(int, manifest.groupby(["subject_id","session_id","label"]).size().unique())), "outer_split_field_read": False, "outer_membership_enumerated": False, "outer_test_used": False})
    payload = {"dataset": config["dataset"], "folds": rows, "outer_split_field_read": False, "outer_membership_enumerated": False, "outer_test_used": False, "data_root": str(__import__("data").stage0_root())}; write_json(OUTPUTS / "DATA_ACCESS_AUDIT.json", payload); persist_split_manifests(); return payload


def smoke(force: bool = False) -> dict[str, Any]:
    device = _device(require_cuda=True); rows = run_smoke(device, force=force); finite = all(np.isfinite(float(row.get("best_eval_BA"))) for row in rows if row.get("best_eval_BA") is not None); payload = {"status": "PASS" if rows and finite else "FAIL", "rows": rows, "paired_initialization_checked": True, "outer_test_used": False, "outer_membership_enumerated": False}; write_json(OUTPUTS / "METHOD_FIDELITY.json", payload); (EXPERIMENT_ROOT / "METHOD_FIDELITY.md").write_text("# Method fidelity\n\nSmoke status: `" + payload["status"] + "`.\n\nB/C are clean-room method-level implementations; they are not exact official-code reproductions.\n", encoding="utf-8"); return payload


def freeze() -> dict[str, Any]:
    config = load_config(); fidelity = json.loads((OUTPUTS / "METHOD_FIDELITY.json").read_text(encoding="utf-8"))
    if fidelity.get("status") != "PASS": raise RuntimeError("method fidelity smoke did not pass")
    code_hashes = {path.name: sha256_file(path) for path in sorted((EXPERIMENT_ROOT / "code").glob("*.py"))}; path = EXPERIMENT_ROOT / "PROTOCOL_FROZEN.json"; payload = {"status": "FROZEN_BEFORE_FULL_RUN", "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "git_sha_before_full_run": git_sha(), "protocol_config": config, "protocol_config_sha256": sha256_file(CONFIG_PATH), "code_sha256": code_hashes, "scientific_gates": {"I1": "mean delta_ID < 0; certified if hierarchical UCB95 < 0", "I2": "valid assignment runs >= frozen minimum, mean L_P > 0 and mean SPL > 0; certified if both LCB95 > 0", "I3": "mean delta_BA_INV < 0; certified if UCB95 < 0", "rescue": "PERSIST - invariant and PERSIST - generic LCB95 > 0"}, "outer_split_field_read": False, "outer_membership_enumerated": False, "outer_test_used": False}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"));
        if old.get("protocol_config_sha256") != payload["protocol_config_sha256"] or old.get("code_sha256") != payload["code_sha256"]: raise RuntimeError("frozen protocol differs; refusing silent re-freeze")
        return old
    write_json(path, payload); return payload


def require_frozen() -> None:
    path = EXPERIMENT_ROOT / "PROTOCOL_FROZEN.json"
    if not path.exists() or json.loads(path.read_text()).get("status") != "FROZEN_BEFORE_FULL_RUN": raise RuntimeError("V1.1 protocol is not frozen")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=["phase0","smoke","freeze","full","audit","functional","rescue","finalize","all"]); parser.add_argument("--force", action="store_true"); args = parser.parse_args()
    if args.phase in {"phase0", "all"}: phase0()
    if args.phase in {"smoke", "all"}: smoke(args.force)
    if args.phase in {"freeze", "all"}: freeze()
    if args.phase in {"full", "all"}: require_frozen(); run_full(_device(True), args.force)
    if args.phase in {"audit", "all"}: require_frozen(); audit_all(args.force)
    if args.phase in {"functional", "all"}: require_frozen(); analyze_all()
    if args.phase in {"rescue", "all"}: require_frozen(); run_eligible_rescues()
    if args.phase in {"finalize", "all"}: require_frozen(); print(json.dumps(finalize(), indent=2))


if __name__ == "__main__": main()
